"""Source-batched execution of public queries; no answers enter a program.

The learned operations are the same as SemanticDecoder.answer. Packing is
limited to one source/observation, so training keeps its optimizer boundaries.
The scalar executor remains the reference and the detailed trace exporter.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .decoders import descriptor_key, query_plan
from .semantic_features import FrozenDict


class SourceProgram:
    def __init__(self, model, inputs):
        self.inputs = tuple(inputs)  # keep identities alive for immutable cache keys
        self.lengths = tuple(len(value['candidates']) for value in inputs)
        self.device = model.descriptors.embedding.weight.device
        self.tensors = {}
        descriptors, descriptor_ids = [], {}

        def descriptor(value):
            key = descriptor_key(value)
            if key not in descriptor_ids:
                descriptor_ids[key] = len(descriptors)
                descriptors.append(value)
            return descriptor_ids[key]

        def intern(values, table, key):
            if key not in table:
                table[key] = len(values)
                values.append(key)
            return table[key]

        plans = [query_plan(value, reuse=model.reuse_query_plans) for value in inputs]
        if model.family != 'structured':
            owners, candidates = [], []
            queries = [descriptor(plan['descriptor']) for plan in plans]
            for index, value in enumerate(inputs):
                candidates.extend(descriptor(c) for c in value['candidates'])
                owners.extend([index] * len(value['candidates']))
            self.pack('queries', queries)
            self.pack('owners', owners)
            self.pack('candidates', candidates)
        else:
            primitives, operations, bindings, means, paths = [], [], [], [], []
            pids, oids, bids, mids, path_ids = {}, {}, {}, {}, {}
            outputs = {'bind': [], 'mean': [], 'path': []}
            positions = {key: [] for key in outputs}

            def primitive(value):
                return intern(primitives, pids, descriptor(value))

            def operation(value):
                return intern(operations, oids, descriptor(value))

            def binding(left, right, op):
                return intern(bindings, bids, (primitive(left), primitive(right), operation(op)))

            offset = 0
            for value, plan in zip(inputs, plans, strict=True):
                ast, candidates = value['ast'], value['candidates']
                if ast['op'] != 'reviewed_choice':
                    raise ValueError('Legacy structured syntax must use the scalar reference path.')
                task, anchor = ast['task'], ast['anchor']
                if task == 'compose':
                    path = intern(paths, path_ids, (primitive(anchor), tuple(operation(s) for s in ast['steps'])))
                    kind, values = 'path', [(path, primitive(c)) for c in candidates]
                elif task == 'binding':
                    kind, values = 'mean', []
                    for row in plan['binding']:
                        if not row:
                            raise ValueError('Empty binding assignment has no grounding computation.')
                        key = tuple(binding(anchor, filler, op) for filler, op in row)
                        values.append(intern(means, mids, key))
                elif task in {'relation', 'identity'}:
                    kind = 'bind'
                    values = [binding(anchor['source'], anchor['target'], op) for op in plan['relations']]
                else:
                    kind, values = 'bind', [binding(anchor, c, ast['operation']) for c in candidates]
                outputs[kind].extend(values)
                positions[kind].extend(range(offset, offset + len(candidates)))
                offset += len(candidates)
            self.pack('primitives', primitives)
            self.pack('operations', operations)
            self.has_bindings = bool(bindings)
            if bindings:
                self.pack('bindings', bindings)
            self.has_means = bool(means)
            if means:
                maximum = max(map(len, means))
                self.pack('mean_indices', [list(row) + [0] * (maximum - len(row)) for row in means])
                self.pack('mean_valid', [[True] * len(row) + [False] * (maximum - len(row)) for row in means], torch.bool)
                self.pack('mean_counts', [len(row) for row in means], model.descriptors.embedding.weight.dtype)
            self.path_depths = []
            if paths:
                self.pack('path_starts', [row[0] for row in paths])
                # Only compose uses dense row-softmax relations. Binding keeps
                # the exact low-rank contraction and invalid-site correction.
                compose_ops = list(dict.fromkeys(op for _, steps in paths for op in steps))
                self.pack('compose_ops', compose_ops)
                compose_ids = {op: i for i, op in enumerate(compose_ops)}
                for depth in range(max(len(row[1]) for row in paths)):
                    active = [(i, compose_ids[steps[depth]]) for i, (_, steps) in enumerate(paths) if depth < len(steps)]
                    self.pack('path_rows_' + str(depth), [i for i, _ in active])
                    self.pack('path_ops_' + str(depth), [op for _, op in active])
                    self.path_depths.append(depth)
            self.output_kinds = [kind for kind in outputs if outputs[kind]]
            concatenated_positions = []
            for kind in self.output_kinds:
                self.pack('output_' + kind, outputs[kind])
                concatenated_positions.extend(positions[kind])
            self.pack('restore_order', np.argsort(concatenated_positions))

        # Flattened bags avoid padding long nested descriptors. padding_idx=0
        # zeros its gradient, while explicit lengths STILL count unknown tokens
        # in the mean denominator, exactly as the original embedding mean did.
        token_ids, offsets, numbers = [], [0], []
        for value in descriptors:
            ids, numeric = model.descriptors.atoms(value, descriptor_key(value))
            token_ids.extend(ids)
            offsets.append(len(token_ids))
            numbers.append(numeric)
        self.pack('tokens', token_ids)
        self.pack('offsets', offsets)
        self.pack('lengths', np.diff(offsets), model.descriptors.embedding.weight.dtype)
        self.pack('numbers', numbers, model.descriptors.embedding.weight.dtype)
        width = max(self.lengths)
        layout, valid, offset = [], [], 0
        for count in self.lengths:
            layout.append(list(range(offset, offset + count)) + [0] * (width - count))
            valid.append([True] * count + [False] * (width - count))
            offset += count
        self.pack('layout', layout)
        self.pack('valid', valid, torch.bool)
        # Bound packed device storage and an allowance for Python plan metadata.
        self.bytes = sum(v.numel() * v.element_size() for v in self.tensors.values())
        self.bytes += sum(len(key.encode('utf-8')) for key in descriptor_ids) + 1024

    def pack(self, name, values, dtype=torch.long):
        self.tensors[name] = torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

    def execute(self, model, observed):
        t = self.tensors
        encoder = model.descriptors
        encoded = F.embedding_bag(t['tokens'], encoder.embedding.weight, t['offsets'],
                                  mode='sum', include_last_offset=True, padding_idx=0)
        encoded = encoder.normalization(encoded / t['lengths'][:, None] + encoder.numeric(t['numbers']))
        if model.family != 'structured':
            query = encoded[t['queries']][t['owners']]
            candidates = encoded[t['candidates']]
            paired = torch.cat((query, candidates, query * candidates), dim=-1)
            flat = model.prior(paired).squeeze(-1)
            if model.family == 'linear':
                flat = flat + (model.query_pair(paired) * observed).sum(-1) / math.sqrt(model.hidden)
            elif model.family == 'mlp':
                condition = model.query_pair(paired)
                flat = flat + model.flexible(torch.cat((observed.expand_as(condition), condition,
                                                       observed * condition), dim=-1)).squeeze(-1)
        else:
            scores = (model.grounding_weight(encoded[t['primitives']]) @ observed.T) / math.sqrt(model.hidden)
            scores = scores.masked_fill(~model.site_mask[None, :], -1e4)
            routes = scores.softmax(-1)
            gates = model.operator(encoded[t['operations']]).tanh()
            left, right = model.left(observed), model.right(observed)
            if self.has_bindings:
                a, b, op = t['bindings'].unbind(-1)
                valid_routes = routes * model.site_mask
                # Moving the operation gates outside the site sum avoids a
                # bindings x sites x rank expansion, with identical algebra.
                l = (valid_routes @ left)[a] * (1 + gates[op, :model.rank])
                r = (valid_routes @ right)[b] * (1 + gates[op, model.rank:])
                invalid = (routes * ~model.site_mask).sum(-1)
                invalid_mass = invalid[a] * routes.sum(-1)[b] + valid_routes.sum(-1)[a] * invalid[b]
                evidence = (routes * scores).sum(-1)
                bound = (r * l).sum(-1) / math.sqrt(model.rank) - 1e4 * invalid_mass
                bound = bound + .5 * (evidence[a] + evidence[b])
            if self.has_means:
                mean = (F.logsigmoid(bound[t['mean_indices']]) * t['mean_valid']).sum(-1) / t['mean_counts']
            if 'path_starts' in t:
                state = routes[t['path_starts']]
                operators = t['compose_ops']
                l = left[None] * (1 + gates[operators, :model.rank, None].transpose(1, 2))
                r = right[None] * (1 + gates[operators, model.rank:, None].transpose(1, 2))
                relations = (l @ r.transpose(1, 2)) / math.sqrt(model.rank)
                relations = relations.masked_fill(~(model.site_mask[:, None] & model.site_mask[None]), -1e4).softmax(-1)
                for depth in self.path_depths:
                    rows, ops = t['path_rows_' + str(depth)], t['path_ops_' + str(depth)]
                    advanced = torch.bmm(state[rows, None, :], relations[ops]).squeeze(1)
                    advanced = advanced * model.site_mask
                    advanced = advanced / advanced.sum(-1, keepdim=True).clamp_min(1e-8)
                    state = state.index_copy(0, rows, advanced)
            blocks = []
            for kind in self.output_kinds:
                indices = t['output_' + kind]
                if kind == 'bind':
                    blocks.append(bound[indices])
                elif kind == 'mean':
                    blocks.append(mean[indices])
                else:
                    blocks.append((state[indices[:, 0]] * scores[indices[:, 1]]).sum(-1))
            flat = torch.cat(blocks)[t['restore_order']]
        return flat[t['layout']].masked_fill(~t['valid'], -torch.inf)


def answer_many(model, observed, inputs):
    """Padded logits, retaining exact candidate order and no cross-query mixing."""
    if not inputs:
        raise ValueError('An observation needs at least one query.')
    supported = model.family != 'structured' or all(v['ast']['op'] == 'reviewed_choice' for v in inputs)
    if not model.source_batched or not supported:
        logits = [model.answer(observed, value)[0] for value in inputs]
        return torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=-torch.inf)
    immutable = all(isinstance(value, FrozenDict) for value in inputs)
    key = tuple(id(value) for value in inputs)
    program = model._source_programs.get(key) if immutable else None
    if program is None:
        program = SourceProgram(model, inputs)
        if immutable and program.bytes <= model.source_program_budget:
            while model._source_programs and (model._source_program_bytes + program.bytes > model.source_program_budget
                                               or len(model._source_programs) >= 2048):
                _, old = model._source_programs.popitem(last=False)
                model._source_program_bytes -= old.bytes
            model._source_programs[key] = program
            model._source_program_bytes += program.bytes
    else:
        if any(a is not b for a, b in zip(program.inputs, inputs, strict=True)):
            raise ValueError('Immutable source query identity changed.')
        model._source_programs.move_to_end(key)
    return program.execute(model, observed)


def choice_targets(examples, device):
    """Supervision stays outside SourceProgram and the neural forward pass."""
    choices = [e['acceptable_indices'] for e in examples]
    for example, values in zip(examples, choices, strict=True):
        if not values or any(i < 0 or i >= len(example['inputs']['candidates']) for i in values):
            raise ValueError('Training requires supported nonempty acceptable-answer indices.')
    width = max(map(len, choices))
    return {
        'indices': torch.tensor([list(v) + [0] * (width - len(v)) for v in choices], device=device),
        'valid': torch.tensor([[True] * len(v) + [False] * (width - len(v)) for v in choices], device=device),
        'weights': torch.tensor([e['weight'] for e in examples], dtype=torch.float32, device=device),
    }


def choice_losses(logits, targets):
    probabilities = logits.log_softmax(-1)
    supported = probabilities.gather(1, targets['indices']).masked_fill(~targets['valid'], -torch.inf)
    return -torch.logsumexp(supported, dim=-1)
