"""Conservative source checks for new groundings, not a semantic accuracy test."""

# These words cannot constitute a one-word English pronoun mention under this
# annotation convention. Quoting a word metalinguistically does not make it a
# pronoun. We deliberately do not use an exhaustive pronoun allowlist: dialect,
# contractions, quantifiers, and pro-adverb classifications require human review.
NON_PRONOUN_WORDS = frozenset("""and or but to of um uh is am are was were be been being
have has had do does did doing get got getting go went going gonna
said say says see saw seen want wants wanted would could should will shall must
can may might looking looks look know think thinks thought the a an in on at
for with from by""".split())


def validate_grounding_spans(catalog, story):
    words = story["words"]
    issues = []
    for mention in catalog.mentions:
        start, end = mention.span.start, mention.span.end
        if mention.form != "pronoun" or end - start != 1:
            continue
        text = words[start]["text"]
        if text.casefold().strip('.,!?:;"') in NON_PRONOUN_WORDS:
            issues.append(f"{mention.id}: pronoun span [{start},{end}) actually covers {text!r}.")
    if issues:
        raise ValueError("Grounding/source mismatch: " + " ".join(issues) +
            " Select the actual referring expression using the printed token indices. "
            "A mention span contains the referring expression, not its verb or the words between references. "
            "Do not merely relabel these words to evade the check; reconstruct the grounded mentions from the text.")
