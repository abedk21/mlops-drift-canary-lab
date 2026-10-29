"""Static, dependency-free HTML report for a synthetic deployment-gate evaluation."""
from __future__ import annotations
import html
import json
from typing import Any


def render(result: dict[str, Any]) -> str:
    scenario = html.escape(str(result["scenario"]))
    gate = result["canary_gate"]
    verdict = html.escape(gate["decision"])
    checks = "\n".join(
        f'<tr><td>{html.escape(str(c["check"]))}</td><td>{"PASS" if c["passed"] else "FAIL"}</td>'
        f'<td>{"n/a" if c["actual"] is None else round(c["actual"], 4)}</td>'
        f'<td>{html.escape(str(c["limit"]))}</td></tr>' for c in gate["checks"])
    f1 = result["candidate_metrics"]["f1"]
    brier = result["candidate_metrics"]["brier"]
    psi = result["feature_drift_psi"]
    bars = "".join(f'<div class="metric"><span>{html.escape(name)}</span><div class="track"><div class="bar" style="width:{min(value/.6,1)*100:.1f}%"></div></div><strong>{value:.3f}</strong></div>' for name,value in psi.items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Synthetic MLOps report</title>
<style>body{{font:15px system-ui,sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;background:#f8fafc;color:#142235}}h1{{margin-bottom:.1rem}}.subtitle{{color:#506172}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1rem}}.tile{{padding:1.1rem;border:1px solid #d5e1e9;background:white;border-radius:12px}}.tile strong{{font-size:1.55rem;display:block}}.metric{{display:flex;gap:1rem;align-items:center;margin:12px 0}}.metric span{{width:145px}}.track{{background:#e2e8f0;border-radius:8px;flex:1;overflow:hidden;height:15px}}.bar{{height:100%;background:#2563eb}}.metric strong{{width:55px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:.7rem;border-bottom:1px solid #e5e7eb;text-align:left}}section{{margin-top:2rem}}code{{background:#e5e7eb;padding:.1rem .3rem}}footer{{margin-top:3rem;color:#53627a}}</style></head><body>
<h1>MLOps Canary & Drift Report</h1><p class="subtitle">Scenario: {scenario}. All records and labels are synthetic; no production deployment occurred.</p>
<div class="grid"><div class="tile">Gate decision<strong>{verdict}</strong></div><div class="tile">Candidate F1<strong>{f1:.3f}</strong></div><div class="tile">Candidate Brier<strong>{brier:.3f}</strong></div><div class="tile">Evaluation size<strong>{result["candidate_metrics"]["n"]}</strong></div></div>
<section><h2>Feature drift · PSI (threshold {gate["policy"]["max_psi_per_feature"]})</h2>{bars}</section>
<section><h2>Explicit release checks</h2><table><thead><tr><th>Check</th><th>Result</th><th>Actual</th><th>Limit</th></tr></thead><tbody>{checks}</tbody></table></section>
<footer>Educational logistic-regression benchmark. Metrics are generated on fabricated support-event traffic. Release decisions are a demo and require separate operational validation.</footer></body></html>'''
