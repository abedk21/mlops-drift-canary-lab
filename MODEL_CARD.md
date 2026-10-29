# Model card — synthetic escalation classifier

## Intended use
Demonstrate a small supervised learning lifecycle for a fictional AI-support-event routing task. It is not designed or validated to decide actual customer escalation, security risk, access, payments, employment, healthcare or eligibility.

## Model
A three-feature logistic regression model optimized by full-batch gradient descent with an L2 weight penalty. Fixed training iteration count, no hyperparameter search, and synthetic Bernoulli labels. Scores are probabilities in `[0,1]`.

## Data and evaluation
Synthetic generator features: `signal_strength`, `context_quality`, `latency_pressure`. Traffic segments: `standard` and `edge`—**operational categories**, not people. Baseline model uses the first 300 records; candidate uses the first 400 of 450 reference records. Evaluation uses 200 new synthetic records generated under a separate random seed. Reference records also supply the PSI baseline distribution. No personal information is present.

## Key metrics
Accuracy, precision, recall, F1 at a fixed 0.5 threshold, Brier score (lower is better), ROC AUC (higher is better), and per-segment recall. AUC is undefined for single-class samples. Segment checks fail closed when there are too few positive labels.

## Release checks (defaults)

- F1 at least 0.58
- F1 regression versus baseline no greater than 0.06
- Brier score at most 0.27
- Each monitored PSI at most 0.25
- Each segment recall at least 0.45 and at least 12 positive examples

These are **illustrative policy cutoffs**, not validated industry standards or official Vertex AI thresholds. An `allow_candidate` decision **does not deploy** a model.

## Risks
Sampling bias, drift misinterpretation, false positives/negatives, hidden correlations, data provenance gaps, optimistic results from synthetic labels, and deployment conditions absent from the simulator. Do not use for real-world decisions or claim production-readiness.
