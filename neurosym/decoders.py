"""Answer-supervised readouts with a typed, spatially local neural executor.

Only public query inputs enter these models. No GraphHistory, answer or evidence
object is accepted. Grounding scores are learned decoding evidence, not labels
for anatomy or estimates of biological connectivity.
"""
import hashlib
import json
import math
import re
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .semantic_queries import validate_ast
from .semantic_features import FrozenDict, FrozenList, freeze_descriptor


_public_serializations = OrderedDict()
_query_plans = OrderedDict()


def query_plan(inputs, *, reuse=True):
    """Compile public immutable syntax only; never cache learned values here."""
    key = id(inputs)
    immutable = isinstance(inputs, FrozenDict)
    if reuse and immutable and key in _query_plans:
        held, plan = _query_plans[key]
        if held is not inputs:
            raise ValueError('Immutable query identity changed.')
        _query_plans.move_to_end(key)
        return plan
    if set(inputs) - {'ast', 'candidates', 'binding_candidates'}:
        raise ValueError('Only public query inputs may enter a decoder.')
    ast, candidates = inputs['ast'], inputs['candidates']
    validate_ast(ast)
    if len(candidates) < 2:
        raise ValueError('Candidate-choice measurement requires alternatives.')
    plan = {'descriptor': freeze_descriptor({'ast': ast, 'binding_candidates': inputs.get('binding_candidates', [])})}
    if ast['op'] == 'reviewed_choice':
        task = ast['task']
        if task == 'binding':
            plan['binding'] = [[(a['filler'], freeze_descriptor({'op': 'role', 'role': a['role'], 'position': a['position']}))
                                for a in c['assignment']] for c in candidates]
            plan['fillers'] = [filler for row in plan['binding'] for filler, _ in row]
        elif task in {'relation', 'identity'}:
            plan['relations'] = [freeze_descriptor({'op': 'event_relation' if task == 'relation' else 'identity',
                'relation': c['label'], 'direction': 'out'}) for c in candidates]
    if reuse and immutable:
        _query_plans[key] = inputs, plan
        if len(_query_plans) > 65536:
            _query_plans.popitem(last=False)
    return plan


def descriptor_key(value, *, representation='json'):
    """Cache only compiler-frozen objects; mutable descriptors are reserialized."""
    key = id(value), representation
    immutable = isinstance(value, (FrozenDict, FrozenList))
    if immutable and key in _public_serializations:
        held, result = _public_serializations[key]
        if held is not value:
            raise ValueError('Immutable descriptor identity changed.')
        _public_serializations.move_to_end(key)
        return result
    result = repr(value) if representation == 'repr' else json.dumps(value, sort_keys=True, ensure_ascii=False)
    if immutable:
        _public_serializations[key] = value, result
        if len(_public_serializations) > 65536:
            _public_serializations.popitem(last=False)
    return result


def descriptor_atoms(value, path="value"):
    return _descriptor_atoms(descriptor_key(value), path)


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
        elif isinstance(item, bool):
            words.append(key + "=" + str(item).lower())
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
        if ast["op"] == "reviewed_choice":
            objects.extend([ast["anchor"], ast["operation"]])
            if ast["task"] in {"relation", "identity"}:
                objects.extend(ast["anchor"].values())
                objects.extend({"op": "event_relation" if ast["task"] == "relation" else "identity",
                                "relation": c["label"], "direction": "out"} for c in inputs["candidates"])
            if ast["task"] == "binding":
                for c in inputs["candidates"]:
                    objects.extend(a["filler"] for a in c["assignment"])
                    objects.extend({"op": "role", "role": a["role"], "position": a["position"]} for a in c["assignment"])
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
        self.batched = True
        self.static = OrderedDict()
        self.packs = OrderedDict()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.packs.clear()
        self.cache = None
        return result

    def atoms(self, value, key):
        if key not in self.static:
            words, numbers = descriptor_atoms(value)
            self.static[key] = ([self.vocabulary.get(w, 0) for w in words] or [0], numbers)
            if len(self.static) > 65536:
                self.static.popitem(last=False)
        self.static.move_to_end(key)
        return self.static[key]

    def many(self, values):
        """Same token mean, numeric projection and LayerNorm in bounded batches.

        Unknown tokens still contribute to the mean's denominator. Only token
        IDs/numbers persist across optimizer steps; learned vectors do not.
        """
        if not self.batched:
            return torch.stack([self(v) for v in values])
        keys = [descriptor_key(v) for v in values]
        cache = self.cache if self.cache is not None else {}
        missing = {k: v for k, v in zip(keys, values, strict=True) if k not in cache}
        missing = list(missing.items())
        for start in range(0, len(missing), 128):
            block = missing[start:start + 128]
            pack_key = tuple(k for k, _ in block)
            if pack_key not in self.packs:
                parts = [self.atoms(v, k) for k, v in block]
                lengths = [len(p[0]) for p in parts]
                ids = [p[0] + [0] * (max(lengths) - n) for p, n in zip(parts, lengths, strict=True)]
                device, dtype = self.embedding.weight.device, self.embedding.weight.dtype
                self.packs[pack_key] = (torch.tensor(ids, device=device),
                    torch.tensor(lengths, device=device, dtype=dtype),
                    torch.tensor([p[1] for p in parts], device=device, dtype=dtype))
                if len(self.packs) > 128:
                    self.packs.popitem(last=False)
            self.packs.move_to_end(pack_key)
            ids, lengths, numbers = self.packs[pack_key]
            # Padding embedding is fixed at zero, as in the scalar path.
            vectors = self.embedding(ids).sum(1) / lengths[:, None]
            encoded = self.normalization(vectors + self.numeric(numbers))
            cache.update((k, v) for (k, _), v in zip(block, encoded.unbind(), strict=True))
        return torch.stack([cache[k] for k in keys])

    def begin_source(self):
        # Reuse gradients only within one source's optimizer step.
        self.cache = {}

    def end_source(self):
        self.cache = None

    def forward(self, value):
        key = descriptor_key(value)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        indices, numbers = self.atoms(value, key)
        device = self.embedding.weight.device
        tokens = torch.tensor(indices, device=device)
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
        self.family, self.shape, self.hidden = family, (() if family == 'prior' else tuple(shape)), hidden
        self.factorized_relations = True
        self.reuse_observation = True
        self.reuse_query_plans = True
        self.source_batched = True
        self.source_program_budget = 128 * 2**20
        self._source_programs = OrderedDict()
        self._source_program_bytes = 0
        self.end_observation()
        sites, times, components = self.shape if family != 'prior' else (0, 0, 0)
        self.descriptors = DescriptorEncoder(vocabulary, hidden)
        self.register_buffer("site_mask", torch.as_tensor([] if family == 'prior' else
                             site_mask if site_mask is not None else np.ones(sites), dtype=torch.bool))
        if family != 'prior' and not self.site_mask.any():
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

    def encode_observation(self, x, *, checked=False):
        self.end_observation()
        if self.family == 'prior':
            return None
        if tuple(x.shape) != self.shape or (not checked and not torch.isfinite(x).all()):
            raise ValueError("Decoder observation shape or values differ from its fitted contract.")
        if self.family in {"linear", "mlp"}:
            return self.global_projection(x.reshape(-1))
        e = F.gelu(torch.einsum("pf,pfh->ph", x.flatten(1), self.local_weight))
        return self.local_shared(e) * self.site_mask[:, None]

    def _apply(self, fn, recurse=True):
        # Packed constants belong to a device/dtype and never contain learned
        # vectors. Rebuild them after to()/double(), without touching weights.
        self._source_programs.clear()
        self._source_program_bytes = 0
        return super()._apply(fn, recurse=recurse)

    def answer_many(self, observed, inputs):
        from .decoder_batch import answer_many
        return answer_many(self, observed, inputs)

    def end_observation(self):
        self._observed = None
        self._observation_primitives = {}
        self._observation_factors = {}
        self._site_factors = None

    def answer(self, observed, inputs, *, capture=False):
        plan = query_plan(inputs, reuse=self.reuse_query_plans)
        ast, candidates = inputs["ast"], inputs["candidates"]
        trace = {"primitive": [], "relations": [], "intermediate": []}
        if self.family != "structured":
            query = self.descriptors(plan['descriptor'])
            candidate_vectors = self.descriptors.many(candidates)
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
        if not self.reuse_observation or self._observed is not observed:
            self.end_observation()
            self._observed = observed
        primitive_cache = self._observation_primitives
        factor_cache, pair_cache = self._observation_factors, {}
        captured_primitives, captured_relations = set(), set()

        def capture_primitive(key, description, scores):
            if capture and key not in captured_primitives:
                trace['primitive'].append({'description': description, 'scores': scores, 'routing': scores.softmax(0)})
                captured_primitives.add(key)

        def primitive(description):
            key = descriptor_key(description, representation='repr')
            if key not in primitive_cache:
                scores = (e @ self.grounding_weight(self.descriptors(description))) / math.sqrt(self.hidden)
                scores = scores.masked_fill(~self.site_mask, -1e4)
                primitive_cache[key] = scores
            capture_primitive(key, description, primitive_cache[key])
            return primitive_cache[key]

        def prime(descriptions):
            descriptions = list({descriptor_key(d, representation='repr'): d for d in descriptions}.values())
            missing = [d for d in descriptions if descriptor_key(d, representation='repr') not in primitive_cache]
            if missing:
                encoded = self.descriptors.many(missing)
                scores = (e @ self.grounding_weight(encoded).T) / math.sqrt(self.hidden)
                scores = scores.masked_fill(~self.site_mask[:, None], -1e4)
                for index, description in enumerate(missing):
                    primitive_cache[descriptor_key(description, representation='repr')] = scores[:, index]
            for description in descriptions:
                capture_primitive(descriptor_key(description, representation='repr'), description, primitive_cache[descriptor_key(description, representation='repr')])

        def factors(operation):
            key = descriptor_key(operation, representation='repr')
            if key not in factor_cache:
                if self._site_factors is None:
                    self._site_factors = self.left(e), self.right(e)
                gates = self.operator(self.descriptors(operation)).tanh()
                factor_cache[key] = (self._site_factors[0] * (1 + gates[:self.rank]),
                                     self._site_factors[1] * (1 + gates[self.rank:]))
            left, right = factor_cache[key]
            if capture and key not in captured_relations:
                trace['relations'].append({'operation': operation, 'left_factor': left,
                                            'right_factor': right, 'scale': math.sqrt(self.rank)})
                captured_relations.add(key)
            return left, right

        def relation(operation):
            key = descriptor_key(operation, representation='repr')
            if key not in pair_cache:
                left, right = factors(operation)
                pair = left @ right.T / math.sqrt(self.rank)
                pair = pair.masked_fill(~(self.site_mask[:, None] & self.site_mask[None]), -1e4)
                pair_cache[key] = pair
            return pair_cache[key]

        def contract(la, ra, operation):
            if not self.factorized_relations:
                return ra @ (la @ relation(operation))
            left, right = factors(operation)
            lm, rm = la * self.site_mask, ra * self.site_mask
            value = (rm @ right) @ (lm @ left) / math.sqrt(self.rank)
            # Retain the dense executor's -1e4 invalid-site values exactly in
            # the algebra; do not assume masked softmax probabilities are zero.
            li, ri = la * ~self.site_mask, ra * ~self.site_mask
            invalid_mass = li.sum() * ra.sum(-1) + lm.sum() * ri.sum(-1)
            return value - 1e4 * invalid_mass

        def bind(left, right, operation):
            ls, rs = primitive(left), primitive(right)
            la, ra = ls.softmax(0), rs.softmax(0)
            return contract(la, ra, operation) + 0.5 * (la @ ls + ra @ rs)

        def bind_many(left, choices, operation):
            # Algebraically identical to repeated bind(), but the anchor-pair
            # product is shared across the complete prefix candidate catalog.
            ls = primitive(left)
            rs = torch.stack([primitive(c) for c in choices])
            la, ra = ls.softmax(0), rs.softmax(-1)
            return (contract(la, ra, operation) + 0.5 * (la @ ls + (ra * rs).sum(-1))).unbind()

        op = ast["op"]
        if op in {"role", "reference", "status", "polarity", "compose"}:
            prime(candidates)
        if op == "reviewed_choice":
            task, anchor = ast["task"], ast["anchor"]
            if task == "compose":
                prime(candidates)
                state = primitive(anchor).softmax(0)
                for step in ast["steps"]:
                    state = state @ relation(step).softmax(-1)
                    state = state * self.site_mask
                    state = state / state.sum().clamp_min(1e-8)
                    if capture:
                        trace["intermediate"].append({"operation": step, "routing": state})
                scores = [state @ primitive(c) for c in candidates]
                latent = state @ e
            elif task == "binding":
                prime(plan['fillers'])
                scores = [torch.stack([F.logsigmoid(bind(anchor, filler, operation)) for filler, operation in row]).mean()
                          for row in plan['binding']]
                latent = primitive(anchor).softmax(0) @ e
            elif task in {"relation", "identity"}:
                scores = [bind(anchor['source'], anchor['target'], operation) for operation in plan['relations']]
                latent = primitive(anchor["source"]).softmax(0) @ e
            else:
                prime(candidates)
                scores = bind_many(anchor, candidates, ast["operation"])
                latent = primitive(anchor).softmax(0) @ e
        elif op == "compose":
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
