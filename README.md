# MLOps Drift & Canary Lab

A complete **local, dependency-free MLOps lifecycle demonstration**: train a real logistic regression classifier with gradient descent, compare it against a baseline on held-out synthetic traffic, compute segment-level quality metrics and feature drift (PSI), and apply explicit deployment-gate rules. Generates a static HTML evaluation dashboard.

**This is not a Vertex AI integration or a deployed inference endpoint.** It demonstrates transferable model-training, evaluation, monitoring and release-policy fundamentals using artificial data.

## What you can show an interviewer

- **Model training:** full-batch gradient descent over three normalized features, with L2 regularization and a reproducible seed.
- **Quality evaluation:** precision, recall, F1, accuracy, Brier score, tie-aware ROC AUC, and a confusion matrix.
- **Slice evaluation:** `standard` and `edge` represent **fictional technical traffic types**, not demographic categories. Detect underperforming segments and enforce minimum positive counts.
- **Data drift:** Population Stability Index (PSI) over fixed normalized feature bins, with finite smoothing for empty bins.
- **Canary promotion:** compare a candidate and baseline using the **same disjoint evaluation population**, enforce quality and drift thresholds, and return `allow_candidate` / `hold_candidate` without deploying anything.
- **Evidence:** documented pass, drift, and model-regression scenarios plus a shareable HTML report.

## Run (Python 3.10+, standard library only)

```bash
python -m mlops_lab --scenario healthy --json-out examples/healthy.json --html-out examples/healthy.html
python -m mlops_lab --scenario shifted --json-out examples/shifted.json --html-out examples/shifted.html
python -m mlops_lab --scenario degraded
python -m unittest discover -s tests -v
```

Open `examples/healthy.html` in a browser. The generated JSON reports contain model weights, split provenance, slice metrics, each PSI score and an explainable list of gate checks.

## Model and monitoring data

The synthetic dataset contains feature values in `[0,1]`: `signal_strength`, `context_quality`, and `latency_pressure`. The target is a fabricated binary escalation indicator. All labels come from a stochastic, documented function of these three features. There are no customer records and no identity or demographic features.

**Three deliberate scenarios**:

| Scenario | What happens | Expected gate result |
| --- | --- | --- |
| `healthy` | Typical evaluation traffic and trained candidate | `allow_candidate` |
| `shifted` | Feature distribution changes sharply, even if raw F1 increases | `hold_candidate` (PSI) |
| `degraded` | Deliberately poor candidate always predicts negative | `hold_candidate` (quality and slices) |

A drift alarm is not inherently proof of bad model performance. The `shifted` scenario illustrates why changes in feature distribution should be reviewed **even if short-term accuracy/recall look strong**.

## Limitations and next steps

This is an educational benchmark, not a statistically robust model release process. The synthetic labels are contemporaneously generated, so realistic label latency, concept drift, label leakage, live request sampling, schema evolution, fairness across protected groups, governance and model registry integration are not represented. PSI thresholds are demonstration policy defaults; they must be calibrated before use in real systems. No actual model endpoint or cloud service is contacted.

Possible extensions: Google Cloud Vertex AI Model Registry and Monitoring adapters, Docker/Cloud Run deployment, MLflow-compatible experiment tracking, time-aware cross-validation, statistical uncertainty intervals, and monitoring with delayed labels.

See [MODEL_CARD.md](MODEL_CARD.md) for constraints and [tests](tests/) for the gate's failure modes.
