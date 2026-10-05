"""Answer-supervised readouts with a typed, spatially local neural executor.

Only public query inputs enter these models. No GraphHistory, answer or evidence
object is accepted. Grounding scores are learned decoding evidence, not labels
for anatomy or estimates of biological connectivity.
"""
import hashlib
import json
import math
import re
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .semantic_queries import validate_ast


def descriptor_atoms(value, path="value"):
    return _descriptor_atoms(json.dumps(value, sort_keys=True, ensure_ascii=False), path)


@lru_cache(maxsize=65536)
def _descriptor_atoms(serialized, path):
    value = json.loads(serialized)
    words, numbers = [], np.zeros(8, dtype=np.float32)
    def visit(item, key):
        if isinstance(item, dict):
            for field in sorted(item):
                words.append("field:" + field)
                visit(item[field], key + "." + field)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, key + f"[{index}]")
        elif isinstance(item, str):
            words.append(key + "=" + item.casefold())
            words.extend("word:" + s for s in re.findall(r"[\w'-]+", item.casefold()))
        elif isinstance(item, (float, int)):
            slot = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(numbers)
            numbers[slot] += np.sign(item) * math.log1p(abs(item)) / 10
        elif item is not None:
            raise ValueError("Unsupported public descriptor value.")
    visit(value, path)
    return tuple(words), tuple(float(n) for n in numbers)


def query_vocabulary(examples):
    tokens = set()
    for example in examples:
        inputs = example["inputs"]
        objects = [inputs["ast"], {"ast": inputs["ast"], "binding_candidates": inputs.get("binding_candidates", [])},
                   *inputs["candidates"], *inputs.get("binding_candidates", [])]
        ast = inputs["ast"]
        objects.extend(ast.get("steps", []))
        for key in ("event", "start", "source", "target", "left", "right"):
            if key in ast:
                objects.append(ast[key])
        objects.extend({"op": "role", "role": a["role"]} for a in ast.get("arguments", []))
        if ast["op"] == "role":
            objects.append({"op": "role", "role": ast["role"]})
        if ast["op"] == "reference":
            objects.append({"mention": ast["mention"]})
        if ast["op"] == "binding":
            objects.append({"label": ast["polarity"]})
        if ast["op"] == "relation":
            objects.extend({"op": "event_relation", "relation": c["label"], "direction": "out", "contexts": ast["contexts"]}
                           for c in inputs["candidates"])
        if ast["op"] == "identity":
            objects.extend({"op": "identity", "relation": c["label"], "contexts": ast["contexts"]} for c in inputs["candidates"])
        for obj in objects:
            tokens.update(descriptor_atoms(obj)[0])
    # Operators and answer-type symbols are language syntax, not test concepts.
    for op in ("reference", "role", "status", "polarity", "binding", "relation", "scope_parent", "argument_event", "event_relation", "identity"):
        tokens.update(descriptor_atoms({"op": op})[0])
    return {token: index + 1 for index, token in enumerate(sorted(tokens))}


class DescriptorEncoder(nn.Module):
    def __init__(self, vocabulary, hidden):
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(len(vocabulary) + 1, hidden, padding_idx=0)
        self.numeric = nn.Linear(8, hidden, bias=False)
        self.normalization = nn.LayerNorm(hidden)
        self.cache = None

    def begin_source(self):
        # Reuse gradients only within one source's optimizer step.
        self.cache = {}

    def end_source(self):
        self.cache = None

    def forward(self, value):
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        words, numbers = descriptor_atoms(value)
        device = self.embedding.weight.device
        tokens = torch.tensor([self.vocabulary.get(w, 0) for w in words] or [0], device=device)
        vector = self.embedding(tokens).mean(0)
        result = self.normalization(vector + self.numeric(torch.as_tensor(numbers, device=device, dtype=vector.dtype)))
        if self.cache is not None:
            self.cache[key] = result
        return result


class SemanticDecoder(nn.Module):
    def __init__(self, family, shape, vocabulary, *, hidden=64, site_mask=None):
        super().__init__()
        if family not in {"prior", "linear", "mlp", "structured"}:
            raise ValueError("Unknown decoder family.")
        self.family, self.shape, self.hidden = family, tuple(shape), hidden
        sites, times, components = self.shape
        self.descriptors = DescriptorEncoder(vocabulary, hidden)
        self.register_buffer("site_mask", torch.as_tensor(site_mask if site_mask is not None else np.ones(sites), dtype=torch.bool))
        if not self.site_mask.any():
            raise ValueError("No observed, trainable support sites.")
        if family != "structured":
            self.prior = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if family in {"linear", "mlp"}:
            self.query_pair = nn.Linear(hidden * 3, hidden, bias=False)
        if family in {"linear", "mlp"}:
            self.global_projection = nn.Linear(sites * times * components, hidden, bias=False)
        if family == "mlp":
            self.flexible = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if family == "structured":
            self.local_weight = nn.Parameter(torch.empty(sites, times * components, hidden))
            nn.init.normal_(self.local_weight, std=1 / math.sqrt(times * components))
            self.local_shared = nn.Linear(hidden, hidden, bias=False)
            self.grounding_weight = nn.Linear(hidden, hidden, bias=False)
            self.rank = min(16, hidden)
            self.left = nn.Linear(hidden, self.rank, bias=False)
            self.right = nn.Linear(hidden, self.rank, bias=False)
            self.operator = nn.Linear(hidden, self.rank * 2)
            self.binding_head = nn.Linear(2, 1)

    def encode_observation(self, x):
        if tuple(x.shape) != self.shape or not torch.isfinite(x).all():
            raise ValueError("Decoder observation shape or values differ from its fitted contract.")
        if self.family == "prior":
            return None
        if self.family in {"linear", "mlp"}:
            return self.global_projection(x.reshape(-1))
        e = F.gelu(torch.einsum("pf,pfh->ph", x.flatten(1), self.local_weight))
        return self.local_shared(e) * self.site_mask[:, None]

    def answer(self, observed, inputs, *, capture=False):
        if set(inputs) - {"ast", "candidates", "binding_candidates"}:
            raise ValueError("Only public query inputs may enter a decoder.")
        ast, candidates = inputs["ast"], inputs["candidates"]
        validate_ast(ast)
        if len(candidates) < 2:
            raise ValueError("Candidate-choice measurement requires alternatives.")
        trace = {"primitive": [], "relations": [], "intermediate": []}
        if self.family != "structured":
            query = self.descriptors({"ast": ast, "binding_candidates": inputs.get("binding_candidates", [])})
            candidate_vectors = torch.stack([self.descriptors(c) for c in candidates])
            paired = torch.cat([query.expand_as(candidate_vectors), candidate_vectors,
                                query * candidate_vectors], dim=-1)
            prior = self.prior(paired).squeeze(-1)
        if self.family == "prior":
            return prior, trace
        if self.family == "linear":
            logits = (self.query_pair(paired) * observed).sum(-1) / math.sqrt(self.hidden) + prior
            if capture:
                trace["latent"] = observed
            return logits, trace
        if self.family == "mlp":
            conditioning = self.query_pair(paired)
            logits = self.flexible(torch.cat([observed.expand_as(conditioning), conditioning,
                                             observed * conditioning], -1)).squeeze(-1) + prior
            if capture:
                trace["latent"] = observed
            return logits, trace

        e = observed
        primitive_cache, pair_cache = {}, {}

        def primitive(description):
            key = repr(description)
            if key not in primitive_cache:
                scores = (e @ self.grounding_weight(self.descriptors(description))) / math.sqrt(self.hidden)
                scores = scores.masked_fill(~self.site_mask, -1e4)
                primitive_cache[key] = scores
                if capture:
                    trace["primitive"].append({"description": description, "scores": scores,
                                                "routing": scores.softmax(0)})
            return primitive_cache[key]

        def prime(descriptions):
            descriptions = list({repr(d): d for d in descriptions}.values())
            encoded = torch.stack([self.descriptors(d) for d in descriptions])
            scores = (e @ self.grounding_weight(encoded).T) / math.sqrt(self.hidden)
            scores = scores.masked_fill(~self.site_mask[:, None], -1e4)
            for index, description in enumerate(descriptions):
                primitive_cache[repr(description)] = scores[:, index]
                if capture:
                    trace["primitive"].append({"description": description, "scores": scores[:, index],
                                               "routing": scores[:, index].softmax(0)})

        def relation(operation):
            key = repr(operation)
            if key not in pair_cache:
                gates = self.operator(self.descriptors(operation)).tanh()
                left = self.left(e) * (1 + gates[:self.rank])
                right = self.right(e) * (1 + gates[self.rank:])
                pair = left @ right.T / math.sqrt(self.rank)
                pair = pair.masked_fill(~(self.site_mask[:, None] & self.site_mask[None]), -1e4)
                pair_cache[key] = pair
                if capture:
                    trace["relations"].append({"operation": operation, "left_factor": left,
                                               "right_factor": right, "scale": math.sqrt(self.rank)})
            return pair_cache[key]

        def bind(left, right, operation):
            ls, rs = primitive(left), primitive(right)
            la, ra = ls.softmax(0), rs.softmax(0)
            return la @ relation(operation) @ ra + 0.5 * (la @ ls + ra @ rs)

        op = ast["op"]
        if op in {"role", "reference", "status", "polarity", "compose"}:
            prime(candidates)
        if op == "compose":
            state = primitive(ast["start"]).softmax(0)
            for step in ast["steps"]:
                matrix = relation(step)
                state = state @ matrix.softmax(dim=-1)
                state = state * self.site_mask
                state = state / state.sum().clamp_min(1e-8)
                if capture:
                    trace["intermediate"].append({"operation": step, "routing": state})
            scores = [state @ primitive(c) for c in candidates]
            latent = state @ e
        elif op == "identity":
            scores = [bind(ast["left"], ast["right"], {"op": "identity", "relation": c["label"], "contexts": ast["contexts"]}) for c in candidates]
            latent = primitive(ast["left"]).softmax(0) @ e
        elif op == "relation":
            scores = [bind(ast["source"], ast["target"], {"op": "event_relation", "relation": c["label"], "direction": "out", "contexts": ast["contexts"]})
                      for c in candidates]
            latent = primitive(ast["source"]).softmax(0) @ e
        elif op == "binding":
            bindings = inputs.get("binding_candidates", [])
            role_scores = [bind(ast["event"], bindings[int(a["candidate"])], {"op": "role", "role": a["role"]})
                           for a in ast["arguments"]]
            if not role_scores:
                raise ValueError("Empty binding query has no grounding computation.")
            # Mean log support is invariant to duplicating identical role evidence.
            joint = torch.stack([F.logsigmoid(s) for s in role_scores]).mean()
            polarity = bind(ast["event"], {"label": ast["polarity"]}, {"op": "polarity"})
            support_logit = self.binding_head(torch.stack([joint, polarity])).squeeze()
            if any(c["label"] not in {"supported", "contradicted"} for c in candidates):
                raise ValueError("This head requires justified binary scope targets.")
            scores = [support_logit if c["label"] == "supported" else -support_logit for c in candidates]
            latent = primitive(ast["event"]).softmax(0) @ e
        elif op == "concept":
            base = primitive({"concept": ast["concept"]})
            evidence = torch.logsumexp(base[self.site_mask], 0) - math.log(int(self.site_mask.sum()))
            scores = [evidence if c["label"] == "supported" else -evidence for c in candidates]
            latent = base.softmax(0) @ e
        else:
            anchor = {"mention": ast["mention"]} if op == "reference" else ast["event"]
            operation = {"op": op}
            if op == "role":
                operation["role"] = ast["role"]
            scores = [bind(anchor, c, operation) for c in candidates]
            latent = primitive(anchor).softmax(0) @ e
        if capture:
            trace["latent"] = latent
            trace["site_mask"] = self.site_mask
        # No direct query-only bypass in the structured model: final evidence
        # must pass through the spatial/coordinate-site grounding computation.
        return torch.stack(scores), trace

    def forward(self, x, inputs, *, capture=False):
        return self.answer(self.encode_observation(x), inputs, capture=capture)


def acceptable_loss(logits, acceptable):
    if not acceptable or any(i < 0 or i >= len(logits) for i in acceptable):
        raise ValueError("Training requires supported nonempty acceptable-answer indices.")
    return -torch.logsumexp(logits.log_softmax(-1)[acceptable], dim=0)


def detached_trace(trace):
    if isinstance(trace, torch.Tensor):
        return trace.detach().cpu().numpy()
    if isinstance(trace, dict):
        return {k: detached_trace(v) for k, v in trace.items()}
    if isinstance(trace, list):
        return [detached_trace(v) for v in trace]
    return trace
