# Falsifiable Hypotheses & Kill Conditions

A hypothesis in this system must be falsifiable: there must be some piece
of evidence that, if found, would prove it wrong. "Churn is caused by
poor onboarding" is falsifiable (you could find onboarding completion
rates are unrelated to churn). "The product isn't good enough" is not —
there's no evidence that could disprove it as stated.

Each hypothesis should be written with:
- **Statement**: a specific, testable claim tied to one branch of the
  issue tree.
- **Kill conditions**: the specific evidence that would falsify it (e.g.
  "if churned users' onboarding completion rate is statistically
  indistinguishable from retained users', this hypothesis is dead").
- **Supporting conditions**: the evidence that would confirm it.

Kill conditions matter more than confirming evidence: a hypothesis that
only ever gets confirming evidence collected for it (never tested against
disconfirming evidence) is not being tested, it's being rationalized.
The researcher and analyst agents should be pointed at kill conditions
specifically, not just "anything supportive."

Good hypotheses are mutually exclusive with each other where possible —
if two hypotheses would both be confirmed by the same piece of evidence,
they're not actually distinguishing between explanations.
