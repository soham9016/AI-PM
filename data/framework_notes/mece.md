# MECE (Mutually Exclusive, Collectively Exhaustive)

MECE is a structuring principle used to break a business problem into an
issue tree: a set of sub-issues that don't overlap (mutually exclusive)
and together cover the whole problem (collectively exhaustive).

Common decompositions:
- Profit = Revenue - Cost (a classic MECE split for "why did profit drop?")
- Revenue = Volume x Price
- A customer funnel: Awareness -> Consideration -> Purchase -> Retention
- Internal vs. external causes; controllable vs. uncontrollable factors

Good MECE trees answer "why" or "how" one level at a time, and each branch
should be independently testable with a hypothesis. A branch that can't be
turned into a falsifiable hypothesis is usually too vague and needs to be
split further. Avoid overlapping branches (e.g. "marketing issues" and
"customer acquisition issues" often overlap) — if two branches could both
explain the same piece of evidence, they aren't mutually exclusive.

A weak issue tree is the most common root cause of unfocused research: if
the top-level split isn't MECE, downstream hypotheses end up either
redundant or leaving gaps uncovered.
