"""Exact local PCA moments and batched GPU projection of actual observations."""
import numpy as np
from scipy import linalg
from .compute import FP64


def story_statistics(values, assignment, *, device='cpu', batch_sites=16):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(assignment) or not np.isfinite(values).all():
        raise ValueError('Invalid real observations for PCA statistics.')
    math = FP64(device)
    result = {'count': np.asarray(len(values), dtype=np.int64)}
    positions = [np.flatnonzero(assignment == i) for i in range(int(assignment.max()) + 1)]
    if not math.gpu:
        for i, p in enumerate(positions):
            local = np.asarray(values[:, p], dtype=np.float64)
            mean = local.mean(0)
            centered = local - mean
            result[f'mean_{i}'], result[f'scatter_{i}'] = mean, centered.T @ centered
        return result
    resident = math.array(values)
    for start in range(0, len(positions), batch_sites):
        group = positions[start:start + batch_sites]
        width = max(map(len, group), default=0)
        if not width:
            for i in range(start, start + len(group)):
                result[f'mean_{i}'], result[f'scatter_{i}'] = np.empty(0), np.empty((0, 0))
            continue
        indices = np.zeros((len(group), width), dtype=np.int64)
        for i, p in enumerate(group):
            indices[i, :len(p)] = p
        local = resident[:, math.torch.as_tensor(indices, device=device)].permute(1, 0, 2)
        mean = local.mean(1)
        local = local - mean[:, None, :]
        scatter = local.transpose(1, 2) @ local
        means, scatters = math.numpy(mean), math.numpy(scatter)
        for j, p in enumerate(group):
            result[f'mean_{start+j}'] = means[j, :len(p)].copy()
            result[f'scatter_{start+j}'] = scatters[j, :len(p), :len(p)].copy()
    return result


def fit_statistics(projector, stories, training_stories, *, device='cpu', eigensolver='cpu'):
    """Merge centered moments using only this fold's training stories."""
    count, means, scatters = 0, [], []
    for record in stories:
        n = int(record['count'])
        if n < 1:
            raise ValueError('Empty training story in PCA moments.')
        if not count:
            means = [record[f'mean_{i}'].copy() for i in range(projector.sites)]
            scatters = [record[f'scatter_{i}'].copy() for i in range(projector.sites)]
        else:
            for i in range(projector.sites):
                delta = record[f'mean_{i}'] - means[i]
                scatters[i] += record[f'scatter_{i}'] + np.outer(delta, delta) * (count * n / (count + n))
                means[i] += delta * (n / (count + n))
        count += n
    if count < 2 or eigensolver not in {'cpu', 'device'}:
        raise ValueError('Invalid training count or eigensolver.')
    projector.training_stories, projector.parameters = list(training_stories), []
    math = FP64(device if eigensolver == 'device' else 'cpu')
    for i in range(projector.sites):
        p = np.flatnonzero(projector.assignment == i)
        scale = np.sqrt(np.maximum(np.diag(scatters[i]) / count, 0))
        use = scale > 1e-8
        p, mean, scale = p[use], means[i][use], scale[use]
        k = min(projector.components, count - 1, len(p))
        if k:
            covariance = scatters[i][np.ix_(use, use)] / scale[:, None] / scale[None, :] / (count - 1)
            if math.gpu:
                eigenvalues, vectors = math.eigh(math.array(covariance))
                eigenvalues, vectors = math.numpy(eigenvalues[-k:]), math.numpy(vectors[:, -k:])
            else:
                eigenvalues, vectors = linalg.eigh(covariance, subset_by_index=(len(p)-k, len(p)-1))
            vectors = vectors[:, eigenvalues > 1e-8][:, ::-1]
            if vectors.shape[1]:
                vectors *= np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
        else:
            vectors = np.empty((len(p), 0))
        projector.parameters.append((p, mean, scale, vectors))
    return projector


def transform_gpu(projector, values, *, device, batch_sites=16):
    math = FP64(device)
    resident = math.array(values)
    result = np.zeros((len(values), projector.sites, projector.components), dtype=np.float32)
    for start in range(0, projector.sites, batch_sites):
        group = projector.parameters[start:start + batch_sites]
        width = max(len(p) for p, _, _, _ in group)
        if not width:
            continue
        indices = np.zeros((len(group), width), dtype=np.int64)
        means, scales = np.zeros((len(group), width)), np.ones((len(group), width))
        vectors = np.zeros((len(group), width, projector.components))
        for i, (p, mean, scale, vec) in enumerate(group):
            indices[i, :len(p)], means[i, :len(p)], scales[i, :len(p)] = p, mean, scale
            vectors[i, :len(p), :vec.shape[1]] = vec
        local = resident[:, math.torch.as_tensor(indices, device=device)].permute(1, 0, 2)
        transformed = ((local - math.array(means)[:, None, :]) / math.array(scales)[:, None, :]) @ math.array(vectors)
        result[:, start:start + len(group)] = math.numpy(transformed).transpose(1, 0, 2)
    return result
