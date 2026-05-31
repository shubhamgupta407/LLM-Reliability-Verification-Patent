# Self-Verifying Large Language Model Systems with Inference-State Consistency Analysis and Multi-Signal Reliability Classification

<p align="center">
  <img src="https://img.shields.io/badge/Status-Patent%20Pending-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Similarity%20Index-14.7%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Institution-LPU%20Punjab-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-AI%20%7C%20LLM%20Safety-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Filed-April%202026-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Claims-22-darkblue?style=for-the-badge" />
</p>

---

## 🧠 What is this?

Large Language Models (LLMs) like GPT-4, Claude, and Gemini are powerful — but they hallucinate. They produce outputs that *sound* confident but are factually wrong, logically broken, or just made up. Existing solutions catch this poorly — they either work post-hoc, require model retraining, or only measure output-level consistency without ever looking at what's happening inside the model during inference.

This invention proposes a **dual-validation architecture** that wraps around *any* LLM and gives every output a quantified trust score *before* it reaches the user or triggers an action — combining independent structural verification with multi-pass inference-state consistency analysis.

**No model retraining. No architecture changes. Plug it in and your LLM becomes self-verifying.**

---

## 🔍 The Core Problem

Standard LLMs produce a softmax probability score — but that score measures *fluency*, not *correctness*. A model can be 97% confident and completely wrong. Existing approaches to fix this:

- **Self-Consistency Prompting** — only looks at output-level majority vote, same model
- **Verifier/Reward Models** — require expensive fine-tuning, fail under distributional shift
- **Confidence Calibration** — post-hoc probability rescaling, no structural verification
- **Deep Ensembles** — multiplicative compute cost, impractical at scale
- **MC Dropout** — requires internal architecture access, breaks on closed-source APIs

None of these combine *independent structural verification* with *internal inference-state consistency analysis* in a deployment-agnostic way. That gap is exactly what this invention addresses.

---

## 🏗️ System Architecture

![System Architecture](figures/fig1_system_architecture.png)

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
  +---------------------------+
  | Reliability Score (R)     |
  | Recommended Disposition   |
  | {ACCEPT / FLAG /          |
  |  REJECT / ESCALATE}       |
  +---------------------------+
```

---

## ⚙️ How It Works

### Channel 1 — Independent Verification (Module 120)

A structurally separate secondary AI model `V` evaluates the primary model's output `O` against the original query `Q`. It checks:
- Logical consistency of the reasoning chain
- Factual coherence against the input context
- Query-output alignment and completeness

Produces a normalized verification score `Sv` in range [0, 1]. Works **zero-shot** — no fine-tuning of the verifier required.

### Channel 2 — Multi-Pass Inference-State Consistency (Module 130)

The primary model is run `N >= 2` times on identical inputs. Signals collected from each pass:

- **Open-weight mode:** hidden layer activations, attention weights, logit distributions
- **Black-box API mode:** output embeddings, token probabilities, semantic similarity across runs

Consistency is computed as:

```
C = 1 / (1 + Var(I))      where Var(I) = E[(I - E[I])^2]
```

High `C` means the model produces stable, consistent internal states across runs — high reliability. Low `C` signals that the model is uncertain even if its output sounds confident.

### Fusion — Confidence Integration Engine (Module 140)

Both signals are fused into a single composite reliability score:

```
R = a * Sv + b * C        where a + b = 1
```

`R` maps to an actionable reliability classification:

| Score Range | Classification | Disposition        |
|-------------|----------------|--------------------|
| 0.85 – 1.00 | HIGH           | Auto-Accept        |
| 0.65 – 0.84 | MEDIUM         | Flag for Review    |
| 0.40 – 0.64 | LOW            | Reject / Re-query  |
| 0.00 – 0.39 | CRITICAL       | Escalate to Human  |

---

## 🌐 Deployment Modes

A key novelty of this architecture is that it operates across all major LLM deployment configurations without modification:

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

- 🏥 **Healthcare** — Suppress low-reliability diagnostic outputs; trigger clinician review before action
- ⚖️ **Legal** — Flag unverifiable citations in contract or case analysis outputs
- 💰 **Finance** — Apply strict reliability thresholds for regulatory reporting and fiduciary advice
- 🔐 **Cybersecurity** — Route uncertain threat analysis to SOC analysts before automated response
- 👔 **Hiring** — Attach reliability disclosures to AI-generated candidate evaluations
- 🤖 **Autonomous Agents** — Gate tool calls and real-world actions behind reliability thresholds

---

## 🔬 Prior Art Differentiation

| Prior Art | Limitation | How This Invention Goes Beyond |
|-----------|------------|-------------------------------|
| Self-Consistency Prompting (Wang et al., 2022) | Same model, output-level only | Structurally independent secondary model + internal inference-state signals |
| Verifier / Reward Models | Dependent on verifier training data quality | Zero-shot verifier; no fine-tuning required |
| Confidence Calibration (Temperature Scaling) | Post-hoc probability adjustment only | Structural reasoning verification, not probability rescaling |
| Deep Ensembles | Multiplicative compute cost | Single primary model + lightweight secondary; no full ensemble overhead |
| MC Dropout | Requires internal architecture access; inapplicable to API models | Explicit black-box mode works with any closed-source API |
| US20250077777A1 | Semantic prompt variations only; no internal state analysis | Multi-pass inference-state consistency + deployment-agnostic reliability fusion |
| US20240370709A1 | Hallucination prevention only; no repeated inference consistency | Dual-channel validation with independent structural verifier fusion |
| US20240378400A1 | Contextual grounding only; no independent verification model | Black-box + open-weight modes with internal transformer-state extraction |
| US20240394512 | Post-generation hallucination detection only | Composite reliability scoring + human escalation + autonomous agent gating |
| US20230259705A1 | Retraining and fact-checking oriented | Runtime deployment-agnostic reliability framework; no base model retraining |

---

## 📋 Patent Information

| Field | Details |
|-------|---------|
| **Title** | Self-Verifying Large Language Model Systems with Inference-State Consistency Analysis and Multi-Signal Reliability Classification |
| **Status** | Invention Disclosure Filed — Patent Pending |
| **Inventor** | Shubham Gupta |
| **Institution** | Lovely Professional University, Punjab — 144411, India |
| **Filed** | April 2026 |
| **Claims** | 22 claims (2 independent + 20 dependent + 1 CRM claim) |
| **Filing Basis** | 35 U.S.C. 101, 102, 103, 112 |
| **Originality Score** | 85.3% (iThenticate v4.2) — 14.7% similarity index |

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

## 👨‍💻 Inventor

| Name | UID | Institution | Contact |
|------|-----|-------------|---------|
| Shubham Gupta | 12306444 | Lovely Professional University, Punjab | aarushjais407@gmail.com |

**GitHub:** [github.com/shubhamgupta407](https://github.com/shubhamgupta407)
**LinkedIn:** [linkedin.com/in/shubhamgupta407](https://linkedin.com/in/shubhamgupta407)

---

## ⚠️ License and Legal

```
Copyright (c) 2026 Shubham Gupta
Lovely Professional University, Punjab, India

All Rights Reserved.

This repository is maintained for documentation, version control, and
prior-art timestamping purposes. No part of this work — including the
system architecture, claims, methods, or figures — may be reproduced,
used, or distributed without the express written permission of the inventor.

Patent Pending. Unauthorized use may constitute patent infringement.
```

---

<p align="center">
  <i>Built at Lovely Professional University · Punjab, India · 2026</i><br/>
  <i>"Making AI say what it means — and mean what it says."</i>
</p>
