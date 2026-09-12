# Samarthan — SIH26094 Functional POC

Samarthan is an engineering proof-of-concept for AI-assisted continuous distress triage and human-led support for PoA victims/complainants.

## Demo workflow

1. **Victim Portal** — calm, minimal check-in experience for text/voice/silent-duress simulation.
2. **AI triage** — lightweight TF-IDF + multinomial logistic regression plus transparent sanity calibration, personal-baseline deviation and channel signals.
3. **Counsellor Console** — priority queue, longitudinal context, explainability and recommended next step.
4. **Human review** — counsellor records an intervention and can tick **Mark this check-in as reviewed**.
5. **Victim notification** — the victim-facing acknowledgement updates to **Counsellor reviewed** and shows the follow-up status.
6. **Audit** — review/action events are recorded.

## Run locally

```text
py -3.14 -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Demo routes

- `/` — role selection
- `/victim` — victim portal
- `/counsellor` — counsellor console
- `/queue` — priority queue
- `/validation` — engineering validation
- `/integrations` — mock government/system adapters
- `/audit` — audit trail
- `/api/health` — health endpoint

## Responsible AI boundaries

This prototype does **not** diagnose mental illness, predict suicide, determine credibility, detect deception, compare victims against one another, or make autonomous intervention decisions.

**AI = triage support. Human = final decision.**

The text model and calibration layer are engineering/demo components trained/evaluated on synthetic examples. Results are **not clinical validation** and must not be used for real-world care or protection decisions.

## Integrations

NHAA 14566, e-Courts, PFMS and Bhashini are represented as mock/adapter interfaces only. No live government data or credentials are included.
