# RICE Scoring

RICE is a prioritization framework for ranking candidate features or
initiatives, used by the `pm` agent when the problem is product-flavored.

RICE score = (Reach x Impact x Confidence) / Effort

- **Reach**: how many users/customers this touches in a given time period
  (e.g. "reaches 4,000 users/quarter").
- **Impact**: how much it moves the needle per user, usually scored on a
  discrete scale — 3 = massive, 2 = high, 1 = medium, 0.5 = low,
  0.25 = minimal.
- **Confidence**: how sure you are about the Reach/Impact estimates,
  expressed as a percentage — 100% = high confidence, 80% = medium,
  50% = low. Low confidence should pull a score down, not be ignored.
- **Effort**: estimated person-time to ship, in a consistent unit
  (e.g. person-months). Larger effort divides the score down.

Higher RICE score = higher priority. RICE is a relative ranking tool, not
an absolute forecast — the value of the score is in comparing candidate
features against each other, not in the raw number itself. Features
should still be sanity-checked against strategic fit before being
greenlit purely by RICE rank.
