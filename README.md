### Self-Verifying Large Language Model Systems with Inference-State Consistency Analysis and Multi-Signal Reliability Classification

<p align="center">
  <img src="https://img.shields.io/badge/Status-Patent%20Pending-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Similarity%20Index-14.7%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Institution-LPU%20Punjab-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-AI%20%7C%20LLM%20Safety-purple?style=for-the-Badge" />
</p>

---

## 🧠 What is this?

Large Language Models (LLMs) like GPT-4, Claude, and Gemini are powerful — but they hallucinate. They produce outputs that *sound* confident but are factually wrong, logically broken, or just made up. Existing solutions catch this poorly.

This model proposes a patented dual-validation architecture that wraps around *any* LLM and gives every output a quantified trust score before it reaches the user or triggers an action.

No model retraining. No architecture changes. Plug it in and your LLM becomes self-verifying.

---

## 🏗️ System Architecture

The system is built around **5 core modules** operating in a dual-channel pipeline:

```
Input Query (Q)
      |
      v
+-------------------------+
|  Primary Inference      |-----> Candidate Output (O)
|  Module 110             |               |
+-------------------------+               |
              |                           |
      +-------+-------+                   |
      v               v                   |
+------------+   +-------------------+    |
| Independent|   | Multi-Pass        |    |
| Verification   | Inference-State   |    |
| Module 120 |   | Consistency       |    |
|            |   | Module 130        |    |
+------------+   +-------------------+    |
  Score (Sv)        Metric (C)            |
      |                  |                |
      +--------+---------+                |
               v                          |
   +------------------------+             |
   | Confidence Integration |             |
   | Engine  Module 140     |             |
   | R = a*Sv + b*C         |             |
   +------------------------+             |
               |                          |
               v                          |
   +------------------------+             |
   | Output Generation      |<------------+
   | Module 150             |
   +------------------------+
               |
               v
  +------------+------------+
  | Reliability Score (R)   |
  | Recommended Disposition |
  | {ACCEPT / FLAG /        |
  |  REJECT / ESCALATE}     |
  +-------------------------+
```

---

## ⚙️ How It Works

### Channel 1 — Independent Verification
A structurally separate secondary AI model `V` evaluates the primary model's output `O` against the original query `Q`. It checks logical consistency, factual coherence, and query-output alignment — producing a normalized verification score `Sv` in range [0, 1].

### Channel 2 — Multi-Pass Inference-State Consistency
The primary model is run `N >= 2` times on identical inputs. Signals collected from each pass:

- **Open-weight mode:** hidden layer activations, attention weights, logit distributions
- **Black-box API mode:** output embeddings, token probabilities, semantic similarity

Consistency is computed as:

```
C = 1 / (1 + Var(I))      where Var(I) = E[(I - E[I])^2]
```

High `C` means the model is stable and confident across runs — high reliability.

### Fusion — Confidence Integration Engine
Both signals are fused into a single composite score:

```
R = a * Sv + b * C        where a + b = 1
```

`R` maps to a reliability classification:

| Score Range | Classification | Disposition        |
|-------------|----------------|--------------------|
| 0.85 – 1.00 | HIGH           | Auto-Accept        |
| 0.65 – 0.84 | MEDIUM         | Flag for Review    |
| 0.40 – 0.64 | LOW            | Reject / Re-query  |
| 0.00 – 0.39 | CRITICAL       | Escalate to Human  |

---

## 🌐 Deployment Modes

| Mode | Description |
|------|-------------|
| **Black-Box API** | Works with any hosted LLM (OpenAI, Anthropic, Gemini) via standard REST — no internal access needed |
| **Open-Weight** | Full internal signal extraction from transformer layers for maximum precision |
| **Autonomous Agent Gate** | Acts as a safety gate in agentic pipelines — withholds actions below reliability threshold |
| **Multimodal** | Handles text, image, audio, video, and tabular output consistency scoring |
| **Edge / On-Prem** | Fully local deployment — no query data leaves the machine |
| **SaaS / Microservices** | Cloud-native deployment with reliability-annotated inference API |

---

## 🏥 Application Domains

Designed specifically for **high-stakes real-world deployment** where hallucination is not acceptable:

- 🏥 **Healthcare** — Suppress low-reliability diagnostic outputs; trigger clinician review
- ⚖️ **Legal** — Flag unverifiable citations in contract or case analysis outputs
- 💰 **Finance** — Apply strict thresholds for regulatory reporting and fiduciary advice
- 🔐 **Cybersecurity** — Route uncertain threat analysis to SOC analysts before automated response
- 👔 **Hiring** — Attach reliability disclosures to AI-generated candidate evaluations

---

## 📂 Repository Contents

```
DualGuard-AI/
|
+-- docs/
|   +-- Patent-Disclosure.docx        # Full invention disclosure (IDF)
|   +-- Plagiarism_Report.docx        # Originality report — 14.7% similarity index
|
+-- figures/                          # Patent figures FIG.1-FIG.8 (in progress)
|   +-- fig1_system_architecture.png
|   +-- fig2_workflow_flowchart.png
|   +-- fig3_signal_extraction.png
|   +-- fig4_variance_comparison.png
|   +-- fig5_confidence_fusion.png
|   +-- fig6_deployment_architecture.png
|   +-- fig7_escalation_workflow.png
|   +-- fig8_agent_gating.png
|
+-- README.md
```

---

## 📜 Patent Information

| Field | Details |
|-------|---------|
| **Title** | Self-Verifying Large Language Model Systems with Inference-State Consistency Analysis and Multi-Signal Reliability Classification |
| **Status** | Invention Disclosure Filed — Patent Pending |
| **Inventors** | Shubham Gupta · Satyam Tomar · Uday Kumar |
| **Institution** | Lovely Professional University, Punjab — 144411, India |
| **Claims** | 22 claims (2 independent + 20 dependent + 1 CRM claim) |
| **Filing Basis** | 35 U.S.C. 101, 102, 103, 112 |
| **Originality Score** | 85.3% (iThenticate v4.2) |

---

## 🔬 Prior Art Differentiation

| Prior Art | Limitation | How DualGuard-AI Goes Beyond |
|-----------|------------|------------------------------|
| Self-Consistency Prompting (Wang et al., 2022) | Same model, output-level only | Uses a structurally independent secondary model + internal state signals |
| Verifier / Reward Models | Dependent on verifier training data quality | No verifier fine-tuning needed; works zero-shot |
| Confidence Calibration (Temperature Scaling) | Post-hoc probability adjustment only | Structural reasoning verification, not probability rescaling |
| Deep Ensembles | Multiplicative compute cost | Single primary model + lightweight secondary; no full ensemble overhead |
| MC Dropout | Requires internal architecture access; inapplicable to API models | Explicit black-box mode works with any API |

---

## ⚠️ License and Legal

```
Copyright (c) 2025 Shubham Gupta, Satyam Tomar, Uday Kumar
Lovely Professional University, Punjab, India

All Rights Reserved.

This repository is maintained for documentation, version control, and
prior-art timestamping purposes. No part of this work — including the
system architecture, claims, methods, or figures — may be reproduced,
used, or distributed without the express written permission of the inventors.

Patent Pending. Unauthorized use may constitute patent infringement.
```

---

## 👨‍💻 Inventors

| Name | UID | Institution |
|------|-----|-------------|
| Shubham Gupta | 12306444 | Lovely Professional University |
| Satyam Tomar | 12315567 | Lovely Professional University |
| Uday Kumar | 12315629 | Lovely Professional University |

---

<p align="center">
  <i>Built at Lovely Professional University · Punjab, India · 2025</i><br/>
  <i>"Making AI say what it means — and mean what it says."</i>
</p>
