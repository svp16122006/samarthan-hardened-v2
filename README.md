# Project Samarthan
## AI-Assisted Continuous Distress Monitoring & Safety Support for PoA Victims

> **SIH26094 · Functional Engineering Proof-of-Concept**
>
> **AI = triage support · Human = final decision**

Project Samarthan is an engineering proof-of-concept for turning a reactive victim-support workflow into a **continuous, human-reviewed support loop**. The prototype accepts a consent-based check-in through a victim-facing interface, analyses text together with available engagement/channel signals and optional engineering audio statistics, compares the resulting signal with the case's historical baseline, and routes the result to an authorised-reviewer workflow.

The prototype is deliberately designed **not** to diagnose mental illness, predict suicide, determine complaint credibility, detect deception, rank victims against one another, or make autonomous intervention decisions.

---

# 1. Evaluator Quick Start

If you have only a few minutes, follow this path:

```text
1. Open the application
        ↓
2. /victim
        ↓
3. Submit a normal, elevated, or urgent check-in
        ↓
4. Open the counsellor workflow
        ↓
5. /queue
        ↓
6. Open a case
        ↓
7. Inspect:
   • Dynamic distress signal
   • Personal baseline
   • Baseline deviation
   • Model confidence
   • Model probabilities
   • Linguistic indicators
   • Engagement / latency signals
   • Optional voice statistics
        ↓
8. Record a human action
        ↓
9. Mark the check-in as reviewed
        ↓
10. Observe victim-facing follow-up status
        ↓
11. /audit
```

### What the POC demonstrates

- Victim-facing check-in workflow
- Text-based ML inference
- Optional WAV audio upload and engineering audio statistics
- Engagement and response-latency signals
- Personal-baseline deviation
- Dynamic distress signal generation
- Explainability information for human review
- Priority review queue
- Longitudinal case history
- Human intervention recording
- Victim-facing review/follow-up status
- Audit logging
- Validation dashboard
- Integration boundaries for NHAA 14566, e-Courts, PFMS and Bhashini

### What the POC does **not** demonstrate

- Live access to government systems
- Clinical validation
- Production authentication/authorisation
- Production deployment security
- A clinically validated mental-health assessment
- Suicide prediction
- Deception or credibility detection
- Autonomous intervention

---

# 2. Problem Context

Victim/complainant support can involve repeated interactions across a long-running case rather than a single point of contact.

The design problem addressed by Samarthan is therefore not simply:

> "Can an AI classify one message?"

It is:

> **Can an existing support workflow continuously surface meaningful changes in distress while keeping the final decision with an authorised human reviewer?**

The POC models this as a closed loop:

```text
Check-in received
       ↓
Signal generated
       ↓
Human review
       ↓
Action / follow-up
       ↓
Later check-in
       ↓
Personal history updated
       ↓
Signal reviewed again
```

The intended design principle is that the system should assist support staff with **triage and continuity**, rather than turn an AI score into an automated decision about a victim.

---

# 3. What Samarthan Does

At a high level:

```text
                 PROJECT SAMARTHAN

 Victim / Complainant
         │
         ├── Text
         ├── Optional voice
         ├── Engagement signals
         ├── Response latency
         └── Silent-duress simulation
                    │
                    ▼
          ┌─────────────────────┐
          │   Signal Analysis   │
          │                     │
          │ Text ML             │
          │ Guardrail signals   │
          │ Channel signals     │
          │ Audio statistics    │
          └──────────┬──────────┘
                     │
                     ▼
          Personal historical context
                     │
                     ▼
          Dynamic distress signal
                     │
                     ▼
          Human review queue
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Human intervention      Continue monitoring
          │                     │
          └──────────┬──────────┘
                     ▼
             Follow-up status
                     │
                     ▼
               Audit trail
```

The victim-facing interface does **not** expose the internal distress score, model probabilities, or counsellor notes.

---

# 4. End-to-End Workflow

## Step 1 — Victim Check-In

The `/victim` interface provides a private-support check-in.

The current POC allows the evaluator to select:

- Case reference
- Preferred language
- Channel
- Text narrative
- Optional voice note
- Engagement signal
- Response latency
- Silent-duress simulation

The interface explicitly tells the user that the system is designed for human follow-up rather than making decisions about them.

---

## Step 2 — Text Enters the ML Pipeline

The current POC's implemented text model is:

```text
Text
 ↓
Tokenisation
 ↓
Unigrams + bigrams
 ↓
TF-IDF representation
 ↓
Multinomial Logistic Regression
 ↓
Class probabilities
 ↓
Predicted class
```

The implementation is dependency-light and written in pure Python so the POC can run without NumPy/SciPy/scikit-learn native extensions.

The relevant implementation is in:

```text
app/ml_core.py
```

---

# 5. How the Implemented Text Model Works

## 5.1 Tokenisation

The model converts input text to lowercase word tokens using a regular-expression tokenizer.

It then constructs:

- 1-gram features
- 2-gram features

For example:

```text
"I feel very unsafe today"
```

can produce features such as:

```text
i
feel
very
unsafe
today

i feel
feel very
very unsafe
unsafe today
```

---

## 5.2 TF-IDF

The POC computes a lightweight TF-IDF representation.

The implementation:

1. Counts terms.
2. Calculates document frequency.
3. Computes inverse document frequency.
4. Applies logarithmic term-frequency scaling.
5. L2-normalises the resulting feature vector.

This converts the text into numerical features that the classifier can process.

---

## 5.3 Multinomial Logistic Regression

The classifier has four classes:

```text
stable
elevated
high
critical
```

The model computes a score for each class and converts the class scores into probabilities using softmax.

The highest-probability class becomes the model's initial predicted label.

The implementation uses gradient-based optimisation with L2 regularisation.

---

# 6. Training Data in the Current POC

The repository contains a deterministic synthetic training generator:

```text
scripts/train_model.py
```

The training examples are organised into four classes:

```text
stable
elevated
high
critical
```

The script starts from controlled seed phrases and combines them with neutral case contexts and temporal modifiers.

Examples of the type of language represented include:

### Stable

```text
I feel safe today.
I feel supported.
I have no immediate concerns.
```

### Elevated

```text
I am worried about the next hearing.
I feel stressed and distracted.
I need more support than before.
```

### High

```text
I cannot sleep and feel overwhelmed.
I feel unsafe and my sleep is very poor.
I am struggling to cope with testimony.
```

### Critical

```text
I am in immediate danger and need help now.
Someone is threatening me right now.
I need protection immediately.
```

The training generator performs controlled augmentation by combining these seed phrases with predefined contexts and modifiers.

### Important limitation

This is **synthetic engineering data**, not a clinical dataset.

Therefore, the reported model metrics demonstrate reproducible software/ML engineering in the POC. They do **not** demonstrate clinical effectiveness, safety, fairness in a real population, or real-world deployment accuracy.

---

# 7. Training / Validation Split

The training script uses a deterministic stratified hold-out split:

```text
Total examples:       320
Training examples:    240
Held-out examples:     80
Split:                 75% / 25%
Seed:                  42
```

The held-out set is not a clinical validation cohort. It is a synthetic engineering evaluation set generated from the same controlled example-generation framework.

---

# 8. Current POC Metrics

The repository's stored validation result is:

| Metric | Result |
|---|---:|
| Accuracy | 96.25% |
| Macro Precision | 96.30% |
| Macro Recall | 96.25% |
| Macro F1 | 96.25% |
| Training examples | 240 |
| Held-out examples | 80 |

Confusion matrix:

```text
                 Predicted
               S   E   H   C
Actual Stable  19  1   0   0
       Elevated 2 18   0   0
       High      0  0  20   0
       Critical  0  0   0  20
```

### How these numbers should be interpreted

**96.25% is not a claim of clinical accuracy.**

The validation page in the POC explicitly describes these results as **held-out engineering validation on synthetic, stratified data**.

For a real pilot, the synthetic examples would need to be replaced by appropriately authorised, ethically governed data and evaluated using a substantially stronger validation design, including language-wise, demographic, channel-wise and temporal analysis as appropriate.

---

# 9. From Model Prediction to Dynamic Distress Signal

The model's text classification is only one component of the current POC's signal-generation pipeline.

The implemented analysis combines:

```text
                 Text model
                     │
                     ▼
              Model class
              + probability
                     │
                     ▼
              Base signal
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
 Text guardrails  Baseline       Channel
                  deviation      signals
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              Final score
                 0–100
                     │
                     ▼
            Final distress label
```

The final score is constrained to the range:

```text
0–100
```

The current implementation maps the result to:

```text
stable
elevated
high
critical
```

The score and label are intended as **triage signals for human review**, not diagnoses.

---

# 10. Transparent Text Guardrail Layer

The POC also includes a transparent lexical sanity layer.

It looks for:

- positive/stability signals
- distress signals
- urgent signals

Examples include language associated with:

```text
unsafe
threatened
cannot sleep
terrified
immediate danger
urgent help
```

This layer can adjust the engineering/demo signal when explicit distress or urgency language is detected.

The code labels this layer as a:

> **Transparent lexical sanity layer for engineering demo; not clinical validation.**

This is intentionally different from claiming that a keyword alone can clinically assess someone's mental state.

---

# 11. Personal Baseline

Samarthan does not intend to compare one victim against another.

Instead, the POC stores a case-level baseline and compares new signals with that case's previous state.

Conceptually:

```text
             Person's own history
                     │
          ┌──────────┴──────────┐
          │                     │
     Historical baseline    Previous check-in
          │                     │
          └──────────┬──────────┘
                     ▼
              Current signal
                     │
                     ▼
              Baseline delta
                     │
                     ▼
          Change in personal state
```

The implementation uses both:

- the stored case baseline
- the previous check-in score, when one exists

to calculate the current baseline deviation component.

This supports the design principle:

> **Track change within the same case rather than ranking victims against each other.**

---

# 12. Engagement and Response-Latency Signals

The POC accepts two additional numerical signals:

```text
engagement
latency
```

These are currently demonstration inputs representing interaction behaviour.

The implementation converts them into bounded signal contributions:

```text
engagement_delta
latency_delta
```

They are then incorporated into the engineering score.

These values should **not** be interpreted as clinically validated behavioural biomarkers.

A production system would require empirical validation before determining whether such signals are appropriate, reliable, culturally robust, and safe to use.

---

# 13. Optional Audio Pathway in the Current POC

The victim check-in accepts an optional WAV upload.

The current implementation does **not** run a Wav2Vec 2.0 model.

Instead, `app/main.py` extracts engineering-level audio statistics from a 16-bit mono WAV:

- RMS amplitude
- Zero-crossing rate
- Duration

The implementation therefore currently demonstrates:

```text
Voice recording
      ↓
WAV parsing
      ↓
RMS
ZCR
Duration
      ↓
Engineering signal contribution
```

This distinction matters.

### Current POC

**Engineering audio statistics**

### Proposed production/pilot direction

A more advanced speech/prosodic model could be evaluated with appropriately governed data, but that is **not represented as an already-trained Wav2Vec 2.0 production model in this repository**.

The README intentionally keeps this distinction clear so that the implementation is not overstated.

---

# 14. Silent-Duress Mechanism

The prototype includes a demonstration of a silent-duress trigger:

```text
DTMF #
```

The API accepts `#` as the demonstration signal and routes it to:

```text
priority_human_review
```

The current API endpoint is:

```text
POST /api/duress
```

The prototype records the event in the audit trail.

### Important boundary

This is a **demonstration mechanism**.

A real IVRS deployment would require:

- authorised channel integration
- accidental-trigger handling
- victim-safe contact protocols
- channel-specific consent
- safety testing
- secure routing
- operational procedures for human response

The POC does not claim that pressing `#` automatically dispatches police or another emergency service.

---

# 15. Human-in-the-Loop Workflow

Samarthan intentionally places a human decision point after AI triage.

```text
AI signal
   ↓
Priority queue
   ↓
Case context
   ↓
Explainability
   ↓
Recommended next step
   ↓
AUTHORISED HUMAN REVIEW
   ↓
Action
   ↓
Follow-up
   ↓
Audit
```

The counsellor can record actions such as:

- Acknowledge and continue monitoring
- Assign counsellor
- Initiate safety planning review
- Escalate to designated official
- Coordinate legal aid / witness support
- Review relocation / protection pathway

The AI recommendation is explicitly presented as a **triage aid, not an autonomous decision**.

---

# 16. Explainability

For each analysed check-in, the POC stores explanation information.

The counsellor case view can display:

### Model information

- ML confidence
- class probabilities
- top linguistic indicators

### Contextual information

- baseline deviation
- engagement contribution
- response-latency contribution
- voice contribution
- silent-duress state

### Guardrail information

- positive signals
- distress signals
- urgent signals

The purpose is to let a human reviewer inspect **why the engineering signal moved** rather than receiving only an unexplained number.

---

# 17. Victim / Counsellor Information Separation

The prototype intentionally shows different information to different roles.

## Victim-facing view

The victim can see:

- check-in acknowledgement
- reference
- support status
- counsellor review status
- follow-up status

The victim does **not** see:

- distress score
- model probabilities
- internal explanation
- counsellor notes

## Counsellor-facing view

The authorised-reviewer demonstration can see:

- dynamic distress signal
- case baseline
- baseline deviation
- model confidence
- model probabilities
- linguistic indicators
- interaction signals
- longitudinal check-ins
- recommended next step
- intervention history
- audit information

This separation is a product/workflow demonstration. It is **not production authentication or access control**.

---

# 18. Closed-Loop Follow-Up

After a counsellor records a review, the check-in can receive:

```text
Monitoring
Follow-up scheduled
Contact requested
Safety review requested
Escalated for authorised support
```

The victim-facing page polls:

```text
GET /api/victim-status/{case_id}/{checkin_id}
```

and can update from:

```text
Awaiting counsellor review
```

to:

```text
Counsellor reviewed
```

The victim therefore receives a simple status rather than the internal AI score.

---

# 19. Longitudinal Case History

Each case can contain multiple check-ins.

The counsellor case view displays the historical sequence, allowing the reviewer to see changes over time rather than treating each message as an isolated event.

Example conceptual history:

```text
Baseline
   │
   ├── Check-in 1 → Elevated
   │
   ├── Check-in 2 → Stable
   │
   ├── Check-in 3 → High
   │
   └── Current → Human review
```

This longitudinal structure is central to the POC's personal-baseline concept.

---

# 20. Integration Architecture

The current repository deliberately separates the core triage engine from external systems.

The integration page identifies these as adapters:

| System | Current POC status | Intended role |
|---|---|---|
| NHAA 14566 | Mock adapter | Case/victim reference and approved communication channel |
| e-Courts | Mock adapter | Case stage / hearing / testimony milestone context |
| PFMS | Mock adapter | Relief/compensation status for follow-up context |
| Bhashini | Adapter interface | Multilingual speech/text routing |

### Critical distinction

There are **no live government credentials or live government data** in this repository.

The prototype demonstrates the intended integration boundaries and workflow. Production use would require authorised APIs/contracts, identity and access controls, governance approval, and appropriate data-handling arrangements.

---

# 21. Proposed Production Architecture

The intended larger architecture can be represented as:

```text
                 EXISTING / APPROVED CHANNELS
        ┌──────────┬─────────┬─────────┬──────────┐
        │ NHAA     │ IVRS    │ SMS     │ WhatsApp │
        │ 14566    │         │         │          │
        └────┬─────┴────┬────┴────┬────┴─────┬────┘
             │           │         │          │
             └───────────┴─────────┴──────────┘
                         │
                         ▼
                SAMARTHAN INTAKE
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Text        Voice      Engagement
             │           │           │
             └───────────┼───────────┘
                         ▼
                 SIGNAL ANALYSIS
                         │
                         ▼
               PERSONAL BASELINE
                         │
                         ▼
              DISTRESS TRIAGE SIGNAL
                         │
                         ▼
                 HUMAN REVIEW QUEUE
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              Intervention    Follow-up
                  │             │
                  └──────┬──────┘
                         ▼
                       Audit
```

e-Courts and PFMS are represented as replaceable contextual adapters rather than hard-coded dependencies.

---

# 22. Data Architecture Concept

Samarthan is designed around a distinction between:

### Case/context information

Examples:

- case reference
- case stage
- authorised milestone information
- support/relief status

### Distress/support information

Examples:

- check-in text
- model outputs
- baseline deviation
- interaction signals
- counsellor review
- follow-up state

The design goal is to prevent distress-triage outputs from becoming an investigative or prosecutorial credibility signal.

The current POC demonstrates this separation primarily through its workflow and UI boundaries; production deployment would require formal data architecture, role-based access controls, encryption, retention policies and governance review.

---

# 23. Responsible AI Boundaries

Samarthan deliberately does **not** claim:

```text
NO DIAGNOSIS
NO SUICIDE PREDICTION
NO DECEPTION DETECTION
NO CREDIBILITY SCORING
NO CROSS-PERSON DISTRESS RANKING
NO AUTONOMOUS INTERVENTION
```

The intended chain is:

```text
AI signal
   ↓
Human review
   ↓
Human decision
   ↓
Recorded action
```

The system is therefore positioned as **decision support for authorised human reviewers**, not an autonomous authority over the victim.

---

# 24. Current Prototype vs Production Proposal

This distinction is important when evaluating the repository.

| Capability | Current repository | Production/pilot requirement |
|---|---|---|
| Victim check-in UI | Implemented | Production UX/security hardening |
| Text ML inference | Implemented | External validation |
| Synthetic training data | Implemented | Authorised governed dataset |
| Personal baseline | Implemented | Pilot validation |
| Engagement signals | Implemented as demo inputs | Empirical validation |
| WAV statistics | Implemented | Validated speech pipeline if adopted |
| Human review queue | Implemented | Production identity/access controls |
| Explainability | Implemented | Human-factors evaluation |
| Audit trail | Implemented | Production immutable/auditable infrastructure |
| NHAA | Mock adapter | Authorised integration |
| e-Courts | Mock adapter | Authorised integration |
| PFMS | Mock adapter | Authorised integration |
| Bhashini | Adapter interface | Production API integration |
| Clinical effectiveness | **Not established** | Formal clinical/field evaluation |
| Production authentication | **Not implemented** | Required |
| Live government data | **Not included** | Requires authorisation |

---

# 25. Demo Data

The application seeds demonstration cases when the local database is empty.

Examples include:

```text
SAM-1042
SAM-1088
SAM-1120
SAM-1147
```

The seed history is intentionally designed to give the counsellor dashboard a longitudinal demonstration story.

The repository therefore contains **demo records**, not real victim records.

The application should not be presented as containing real government or clinical data.

---

# 26. Repository Structure

```text
samarthan-hardened-v2/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ml_core.py
│   │
│   ├── templates/
│   │   ├── audit.html
│   │   ├── integrations.html
│   │   ├── counsellor.html
│   │   ├── help.html
│   │   ├── dashboard.html
│   │   ├── victim_history.html
│   │   ├── case.html
│   │   ├── checkin.html
│   │   ├── received.html
│   │   ├── queue.html
│   │   └── validation.html
│   │
│   └── static/
│       └── style.css
│
├── models/
│   ├── model.pkl
│   └── metrics.json
│
├── scripts/
│   └── train_model.py
│
├── tests/
│   └── test_core.py
│
├── Dockerfile
├── requirements.txt
├── run_windows.bat
├── .gitignore
└── README.md
```

---

# 27. Important Source Files

## `app/main.py`

Application layer containing:

- FastAPI routes
- database initialisation
- case/check-in storage
- audio-stat extraction
- signal analysis
- victim workflow
- counsellor workflow
- queue
- audit
- validation page
- integration adapters
- API endpoints

## `app/ml_core.py`

Dependency-light ML implementation containing:

- tokenizer
- TF-IDF vectoriser
- multinomial logistic regression
- probability prediction
- top-term extraction
- model serialisation

## `scripts/train_model.py`

Creates the synthetic training examples, performs the deterministic stratified split, trains the model and writes:

```text
models/model.pkl
models/metrics.json
```

## `models/model.pkl`

Serialised trained POC model.

## `models/metrics.json`

Stored evaluation metadata and confusion matrix.

## `tests/test_core.py`

Basic unit tests covering:

- risk-label ordering
- human-action mapping

---

# 28. Installation

## Requirements

The project is designed for:

```text
Python 3.14
```

Install dependencies from:

```text
requirements.txt
```

The main dependencies are:

- FastAPI
- Uvicorn
- Jinja2
- python-multipart

---

# 29. Run on Windows

Create a virtual environment:

```powershell
py -3.14 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The repository also contains:

```text
run_windows.bat
```

for Windows-oriented startup.

---

# 30. Train the Model

If the model file does not already exist:

```powershell
python scripts\train_model.py
```

This generates:

```text
models/model.pkl
models/metrics.json
```

The application also contains logic to generate the model if it is missing.

---

# 31. Demo Routes

| Route | Purpose |
|---|---|
| `/` | Role-selection / entry dashboard |
| `/victim` | Victim check-in |
| `/victim/help` | Victim help |
| `/victim/history` | Victim support history |
| `/counsellor` | Counsellor console |
| `/queue` | Human review queue |
| `/case/{case_id}` | Detailed human review |
| `/validation` | ML validation lab |
| `/integrations` | Integration boundaries |
| `/audit` | Audit trail |
| `/api/health` | Health endpoint |
| `/api/cases` | Case data endpoint |
| `/api/duress` | Silent-duress demonstration endpoint |
| `/api/victim-status/{case_id}/{checkin_id}` | Victim-facing review/follow-up status |

---

# 32. Recommended Judge Demo

## A. Start at the role-selection page

Open:

```text
/
```

Explain:

> "Samarthan has two intentionally separated workflows: a victim-facing check-in and a counsellor-facing human-review workspace."

---

## B. Open the victim portal

Open:

```text
/victim
```

Select a demo case.

Use one of the provided stable/elevated/urgent demo narratives.

Submit the check-in.

The victim-facing page should show that the response was received.

It does not reveal the internal distress score.

---

## C. Move to the counsellor console

Open:

```text
/queue
```

Show that the latest signals are surfaced for human review.

Select a case.

---

## D. Explain the case screen

Point out:

```text
Personal baseline
Current dynamic distress signal
Baseline deviation
ML confidence
Class probabilities
Linguistic indicators
Engagement contribution
Latency contribution
Voice contribution
Recommended next step
```

Then explain:

> "The important part is that the model does not close the case. It gives the reviewer evidence to inspect before the reviewer chooses an action."

---

## E. Demonstrate the longitudinal component

Show multiple historical check-ins.

Explain:

> "The system is interested in change over the same person's history rather than comparing one victim against another."

---

## F. Record a human action

Choose a human action and optionally add reviewer notes.

Mark the check-in as reviewed.

---

## G. Return to the victim-facing status

Open the corresponding received/status page.

Show that the victim-facing state can change to:

```text
Counsellor reviewed
```

while the internal score and counsellor notes remain hidden.

---

## H. Show the audit trail

Open:

```text
/audit
```

Show that AI analysis and human actions are recorded.

---

## I. Show the validation lab

Open:

```text
/validation
```

Explain the metrics only in their correct context:

> "This is reproducible engineering validation on synthetic data, not clinical validation."

---

## J. Show integrations last

Open:

```text
/integrations
```

Explain:

> "These are deliberately mock/adapter boundaries. No live government credentials or data are bundled with the POC."

---

# 33. API Examples

## Health

```http
GET /api/health
```

Returns the service status and identifies the current POC model as:

```text
TF-IDF + Logistic Regression
```

---

## Cases

```http
GET /api/cases
```

Returns the locally stored case records used by the demonstration application.

---

## Victim Status

```http
GET /api/victim-status/{case_id}/{checkin_id}
```

Returns whether the check-in has been reviewed and its follow-up status.

---

## Silent Duress

```http
POST /api/duress
```

The demonstration accepts:

```text
#
```

and records a priority human-review event.

Other signals are rejected by the demonstration endpoint.

---

# 34. Testing

Run:

```powershell
python -m unittest discover -s tests
```

The current tests cover basic core behaviour, including:

```text
stable < critical
```

in the risk ordering and ensuring the critical action mapping contains a human-review action.

The current test suite is intentionally small and should not be interpreted as comprehensive production assurance.

---

# 35. Docker

The repository includes a Dockerfile based on:

```text
python:3.14-slim
```

The container:

1. Installs `requirements.txt`
2. Copies the application
3. Starts Uvicorn
4. Uses the `PORT` environment variable when provided

Typical container command:

```text
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

# 36. Security and Production Hardening

The current POC demonstrates the **workflow concepts**, not a production security boundary.

Before deployment, a real implementation would need, at minimum:

- strong authentication
- role-based authorisation
- session management
- encryption in transit
- encryption at rest
- secure secret management
- audit-log protection
- input validation
- upload validation
- malware/content scanning for uploads
- rate limiting
- secure API gateways
- data retention/deletion controls
- least-privilege access
- incident response
- privacy impact assessment
- formal governance review

The current repository intentionally does not claim that these production controls are complete.

---

# 37. Model and Safety Limitations

The current POC has several limitations that should be understood by evaluators.

### Synthetic data

The training/held-out examples are synthetic and controlled.

### No clinical validation

The metrics do not establish clinical validity.

### No real-world effectiveness

The prototype has not established that its signal improves outcomes in real victim-support settings.

### Audio limitation

The current audio implementation calculates basic engineering statistics rather than running a clinically validated speech-emotion model.

### Demo engagement signals

Engagement and latency are currently supplied as demonstration signals.

### Rule-based guardrails

The lexical sanity layer is transparent engineering logic, not a clinical assessment.

### Production access controls

The role separation in the UI is a workflow demonstration, not production authentication.

### External integrations

Government systems are represented by mock/adapter interfaces.

---

# 38. Cold-Start Consideration

Personal-baseline monitoring naturally has a cold-start problem.

For a new case:

```text
No historical baseline
        ↓
Initial contextual signal
        ↓
Human review
        ↓
History accumulates
        ↓
Personal baseline becomes more informative
```

The current POC seeds baseline values for demonstration cases.

A production implementation should define a conservative cold-start policy and validate it during pilot deployment.

---

# 39. What Would Be Required for Pilot Validation?

A real pilot should not simply deploy the synthetic model and measure accuracy.

It should evaluate questions such as:

### Model performance

- precision
- recall
- F1
- calibration
- false-positive rate
- false-negative rate

### Robustness

- language
- channel
- audio quality
- transcription quality
- case stage
- temporal drift

### Human factors

- reviewer workload
- reviewer agreement
- time-to-review
- usefulness of explanations
- inappropriate-alert rate

### Safety

- missed urgent situations
- inappropriate escalation
- unsafe communication
- accidental silent-duress triggers

### Privacy/governance

- access control
- retention
- consent
- data minimisation
- auditability

The goal should be to establish whether the system is useful and safe **before** relying on it in real support operations.

---

# 40. Production Learning Strategy

The current POC should not be interpreted as continuously retraining itself online.

A safer production-oriented learning loop would be:

```text
Approved / governed feedback
          ↓
Data quality review
          ↓
Offline retraining
          ↓
Validation
          ↓
Bias / safety evaluation
          ↓
Human approval
          ↓
Versioned model release
```

The deployed model should therefore be treated as a versioned artefact rather than an uncontrolled online-learning system.

---

# 41. Why the Architecture Uses Human Review

The central architectural principle is:

```text
AI detects a signal
        ≠
AI decides what happens
```

Instead:

```text
AI
 ↓
surfaces signal
 ↓
explains contributing information
 ↓
prioritises review
 ↓
human checks context
 ↓
human decides
 ↓
system records action
```

This is especially important because a distress signal can have multiple explanations and because automated classification on its own cannot establish what intervention is appropriate.

---

# 42. Integration Philosophy

Samarthan is designed to sit **on top of existing channels**, rather than requiring a new victim-side application or dedicated hardware.

Conceptually:

```text
Existing communication channels
             ↓
       Samarthan intake
             ↓
      AI-assisted triage
             ↓
       Human support
```

The current POC uses its own web interface to demonstrate the workflow, while the integration layer represents how approved external channels could feed the same core system.

---

# 43. Multilingual Direction

The architecture includes a Bhashini adapter/interface for multilingual speech/text routing.

In the current POC, this is an **integration boundary/demo interface**, not evidence that the distress model has been clinically validated across all Indian languages.

A production pilot should perform language-wise evaluation before relying on model outputs across different languages and dialects.

---

# 44. Ethical Design Principles

Samarthan is built around several constraints:

1. **Support, not diagnosis**
2. **Triage, not autonomous decision-making**
3. **Personal history, not cross-victim ranking**
4. **Human review before intervention**
5. **Victim-facing privacy of internal AI outputs**
6. **Auditability of AI and human actions**
7. **Explicit distinction between prototype and production integrations**
8. **Synthetic-data limitations are disclosed**
9. **No use of the system as a credibility/deception detector**

---

# 45. Project Status

### Current status

**Functional engineering POC**

Implemented in this repository:

- Web application
- Victim workflow
- Counsellor workflow
- Text ML inference
- Synthetic model training
- Personal-baseline calculation
- Dynamic signal generation
- Optional audio statistics
- Explainability
- Human actions
- Follow-up status
- Audit logging
- Validation page
- Integration adapter demonstrations
- Docker configuration
- Basic automated tests

### Not yet established

- Clinical effectiveness
- Real-world safety
- Government production integration
- Production authentication/authorisation
- Field performance
- Population-level fairness
- Production-scale reliability

---

# 46. Suggested Evaluation Narrative

If an evaluator asks:

### "What is actually innovative here?"

The answer is not simply "AI".

The POC combines:

```text
Existing channels
      +
Longitudinal personal baseline
      +
Multimodal/contextual signals
      +
Explainable triage
      +
Human review
      +
Follow-up
      +
Audit
```

into one continuous support workflow.

### "Does the AI make the decision?"

No.

It produces a triage signal and supporting information. The human reviewer decides the next action.

### "Is the model clinically validated?"

No.

The current metrics are from a synthetic engineering dataset and are explicitly not clinical validation.

### "Are NHAA/e-Courts/PFMS actually connected?"

No.

They are represented as mock/adapter interfaces in this POC. Production integration would require authorised access and governance.

### "Does the victim see the AI score?"

No.

The victim-facing workflow intentionally hides internal distress scores, probabilities and counsellor notes.

---

# 47. Final Architecture in One View

```text
                         PROJECT SAMARTHAN
             AI-ASSISTED CONTINUOUS SUPPORT LAYER

 ┌──────────────────────────────────────────────────────────┐
 │                  EXISTING / APPROVED CHANNELS             │
 │                                                          │
 │ NHAA 14566 · IVRS · SMS · WhatsApp · Web · Optional App │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ SAMARTHAN INTAKE  │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
            TEXT            VOICE         ENGAGEMENT
              │               │                │
              ▼               ▼                ▼
           TF-IDF       Audio statistics    Channel
              │                            signals
              └───────────────┬────────────────┘
                              ▼
                    ┌───────────────────┐
                    │ SIGNAL GENERATION │
                    │                   │
                    │ ML probabilities  │
                    │ Guardrails        │
                    │ Baseline delta    │
                    │ Context signals   │
                    └─────────┬─────────┘
                              │
                              ▼
                   DYNAMIC DISTRESS SIGNAL
                              │
                              ▼
                    ┌───────────────────┐
                    │ HUMAN REVIEW QUEUE│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ CASE EXPLANATION  │
                    │                   │
                    │ Baseline          │
                    │ Probabilities     │
                    │ Indicators        │
                    │ Context           │
                    └─────────┬─────────┘
                              │
                              ▼
                     AUTHORISED HUMAN
                         DECISION
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
            Monitoring    Follow-up      Escalation
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                       AUDIT + HISTORY
                              │
                              ▼
                       NEXT CHECK-IN
```

---

# 48. Bottom Line

Project Samarthan is currently a **working engineering POC**, not a clinically validated mental-health system.

Its demonstrated contribution is the workflow:

> **existing-channel check-in → machine-assisted signal generation → personal-baseline context → explainable human review → intervention/follow-up → audit → longitudinal monitoring**

The repository intentionally separates what is implemented today from what would require authorised integration, governed data, production security and real-world validation.

That distinction is part of the design.

---

## Links

### Live Prototype

Add the final deployed prototype URL here:

```text
https://samarthan-demo.getvoroa.com/
```

### Source Repository

```text
https://github.com/svp16122006/samarthan-hardened-v2
```

### Recommended Demo Video

Add the final recorded walkthrough here once available.

---

## License / Usage

This repository is an SIH engineering proof-of-concept. It should not be used to make real-world clinical, protection, legal, investigative, or other high-impact decisions.

**AI = triage support. Human = final decision.**
