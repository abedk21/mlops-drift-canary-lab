"""Pure-Python synthetic binary classifier, evaluation and deployment-gate building blocks.

This project is a teaching reference: it does not connect to Vertex AI or assess
real people's characteristics. Data and traffic segments are fabricated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Iterable

FEATURES = ("signal_strength", "context_quality", "latency_pressure")


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def samples(n: int, seed: int, *, shifted: bool = False) -> list[dict[str, Any]]:
    """Generate documented, reproducible artificial training or monitoring traffic."""
    if not 1 <= n <= 10000:
        raise ValueError("n must be in [1, 10000]")
    rng = random.Random(seed)
    records = []
    for idx in range(n):
        segment = "edge" if rng.random() < .30 else "standard"
        signal = rng.uniform(.05, .9)
        quality = rng.uniform(.15, 1)
        latency = rng.uniform(0, .85)
        if segment == "edge":
            quality = clamp(quality - .13)
            latency = clamp(latency + .12)
        if shifted:
            signal = clamp(signal + .30)
            quality = clamp(quality - .31)
            latency = clamp(latency + .28)
        # Synthetic signal generated from a fixed function plus stochastic noise.
        latent = -2.6 + 3.3 * signal + 2.0 * (1 - quality) + 1.7 * latency
        probability = sigmoid(latent)
        label = int(rng.random() < probability)
        records.append({"id": f"SYN-{seed}-{idx:04d}", "segment": segment,
                        "features": {"signal_strength": signal, "context_quality": quality,
                                     "latency_pressure": latency}, "label": label})
    return records


@dataclass(frozen=True)
class Model:
    weights: tuple[float, float, float]
    bias: float
    trained_examples: int
    training_method: str = "batch_gradient_descent_logistic_regression"

    def predict(self, features: dict[str, float]) -> float:
        values = [float(features[k]) for k in FEATURES]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("Expected finite normalized features in [0, 1]")
        return sigmoid(self.bias + sum(x*w for x, w in zip(values, self.weights)))

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = list(FEATURES)
        d["weights"] = list(self.weights)
        return d


def fit(training: list[dict[str, Any]], epochs: int = 1100, lr: float = 0.35, l2: float = 0.001) -> Model:
    """Actual batch gradient descent over the synthetic labels (no ML dependencies)."""
    if len(training) < 20:
        raise ValueError("At least 20 training records required")
    if not 1 <= epochs <= 10000 or not 0 < lr <= 1 or not 0 <= l2 <= 1:
        raise ValueError("Invalid optimization parameters")
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    n = len(training)
    prepared = []
    for row in training:
        y = row["label"]
        if type(y) is not int or y not in (0, 1):
            raise ValueError("Training labels must be 0 or 1")
        x = [float(row["features"][k]) for k in FEATURES]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in x):
            raise ValueError("Bad feature value")
        prepared.append((x, y))
    for _ in range(epochs):
        gradients = [0.0] * len(FEATURES)
        bias_grad = 0.0
        for x, y in prepared:
            error = sigmoid(bias + sum(a*b for a,b in zip(weights,x))) - y
            bias_grad += error
            for i in range(len(FEATURES)):
                gradients[i] += error * x[i]
        for i in range(len(FEATURES)):
            weights[i] -= lr * (gradients[i] / n + l2 * weights[i])
        bias -= lr * bias_grad / n
    return Model(tuple(weights), bias, n)


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Tie-aware rank AUC; None for a single-class evaluation set."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    total_positive_ranks = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = ((i + 1) + j) / 2
        total_positive_ranks += rank * sum(label for _, label in pairs[i:j])
        i = j
    return (total_positive_ranks - pos*(pos + 1)/2) / (pos*neg)


def score(model: Model, rows: list[dict[str, Any]], threshold: float = .5) -> dict[str, Any]:
    if not rows or not 0 < threshold < 1:
        raise ValueError("Need data and a threshold in (0,1)")
    predictions = [model.predict(row["features"]) for row in rows]
    labels = [row["label"] for row in rows]
    tp = sum(p >= threshold and y == 1 for p,y in zip(predictions,labels))
    fp = sum(p >= threshold and y == 0 for p,y in zip(predictions,labels))
    fn = sum(p < threshold and y == 1 for p,y in zip(predictions,labels))
    tn = sum(p < threshold and y == 0 for p,y in zip(predictions,labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    brier = sum((p-y)**2 for p,y in zip(predictions,labels))/len(rows)
    by_segment = {}
    for segment in sorted(set(row["segment"] for row in rows)):
        subset = [(y,p) for row,y,p in zip(rows,labels,predictions) if row["segment"]==segment]
        positives = sum(y for y,p in subset)
        true_positives = sum(y==1 and p>=threshold for y,p in subset)
        by_segment[segment] = {"n":len(subset), "positives":positives,
                               "recall": true_positives / positives if positives else None,
                               "brier":sum((p-y)**2 for y,p in subset)/len(subset)}
    return {"n":len(rows), "confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},
            "accuracy":(tp+tn)/len(rows), "precision":precision, "recall":recall,
            "f1":f1, "brier":brier, "auc":roc_auc(labels,predictions), "segments":by_segment}


def psi(reference: Iterable[float], observed: Iterable[float], bins: int = 10) -> float:
    """Population Stability Index on fixed [0,1] bins, epsilon-smoothed."""
    a,b=list(reference),list(observed)
    if not a or not b or bins < 2 or bins > 40:
        raise ValueError("Need nonempty distributions and 2..40 bins")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in a+b):
        raise ValueError("Invalid normalized feature distribution")
    ac,bc=[0]*bins,[0]*bins
    for value in a:
        ac[min(int(value*bins),bins-1)]+=1
    for value in b:
        bc[min(int(value*bins),bins-1)]+=1
    # Additive smoothing makes PSI finite when a bin is initially empty.
    eps = 0.0001
    return sum(((y+eps)/(len(b)+bins*eps)-(x+eps)/(len(a)+bins*eps)) *
               math.log(((y+eps)/(len(b)+bins*eps))/((x+eps)/(len(a)+bins*eps)))
               for x,y in zip(ac,bc))


def drift(reference: list[dict[str, Any]], observed: list[dict[str, Any]]) -> dict[str, float]:
    return {key:psi((r["features"][key] for r in reference),
                    (r["features"][key] for r in observed)) for key in FEATURES}


@dataclass(frozen=True)
class Policy:
    min_f1: float = .58
    max_f1_regression: float = .06
    max_psi_per_feature: float = .25
    max_brier: float = .27
    min_segment_recall: float = .45
    min_segment_positives: int = 12


def canary_gate(baseline: dict[str, Any], candidate: dict[str, Any], feature_drift: dict[str, float],
                policy: Policy = Policy()) -> dict[str, Any]:
    """Explicit policy-based decision; does not deploy a model."""
    checks = []
    checks.append({"check":"candidate_f1", "passed":candidate["f1"]>=policy.min_f1,
                   "actual":candidate["f1"], "limit":policy.min_f1})
    delta = baseline["f1"] - candidate["f1"]
    checks.append({"check":"f1_regression", "passed":delta<=policy.max_f1_regression,
                   "actual":delta, "limit":policy.max_f1_regression})
    checks.append({"check":"brier_score", "passed":candidate["brier"]<=policy.max_brier,
                   "actual":candidate["brier"], "limit":policy.max_brier})
    for name,value in feature_drift.items():
        checks.append({"check":f"psi_{name}", "passed":value<=policy.max_psi_per_feature,
                       "actual":value, "limit":policy.max_psi_per_feature})
    for segment, metrics in candidate["segments"].items():
        enough = metrics["positives"] >= policy.min_segment_positives
        passed = enough and metrics["recall"] is not None and metrics["recall"]>=policy.min_segment_recall
        checks.append({"check":f"segment_recall_{segment}", "passed":passed,
                       "actual":metrics["recall"], "positives":metrics["positives"],
                       "min_positives":policy.min_segment_positives,"limit":policy.min_segment_recall})
    return {"decision":"allow_candidate" if all(c["passed"] for c in checks) else "hold_candidate",
            "checks":checks, "policy":asdict(policy), "actual_deployment_performed":False}


def experiment(scenario: str = "healthy", seed: int = 13) -> dict[str, Any]:
    if scenario not in {"healthy", "shifted", "degraded"}:
        raise ValueError("Choose healthy, shifted or degraded")
    reference=samples(450,seed)
    observation=samples(200,seed+1,shifted=scenario=="shifted")
    baseline=fit(reference[:300])
    candidate=fit(reference[:400])
    if scenario=="degraded":
        candidate=Model((0,0,0),-8,candidate.trained_examples,"intentionally_bad_synthetic_candidate")
    baseline_metrics=score(baseline,observation)
    candidate_metrics=score(candidate,observation)
    feature_drift=drift(reference,observation)
    gate=canary_gate(baseline_metrics,candidate_metrics,feature_drift)
    return {"scenario":scenario,"synthetic_data":True,
            "provenance":{"generator_seed":seed,"reference_n":len(reference),"observation_n":len(observation),
                          "training_rows":candidate.trained_examples,"split":"disjoint generated seeds"},
            "candidate_model":candidate.as_dict(), "baseline_metrics":baseline_metrics,
            "candidate_metrics":candidate_metrics,"feature_drift_psi":feature_drift,"canary_gate":gate}
