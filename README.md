# Customer Support AI Agent and Evaluation System

## Automated triage, grounded reply drafting, and evaluation harness for @SpotifyCares

This repository implements an end-to-end customer support agent and evaluation
harness, built on real customer support interactions from the Customer Support
on Twitter dataset (`twcs.csv`). The agent classifies incoming messages into a
7-intent taxonomy, drafts replies grounded in historical brand resolutions,
and decides whether each message should be auto-handled or escalated to a
human, with a stated reason.

## Deliverables Matrix

| Deliverable | Location | Description |
|---|---|---|
| 1. Runnable pipeline (under 15 min reproduction) | `prepare_data.py`, `baselines.py`, `agent.py`, `evaluate.py` | Full pipeline reproducible in under 15 minutes. Full live run: 8 to 12 minutes. Fast live smoke test: about 90 seconds. Offline mode: about 5 seconds. |
| 2. Golden evaluation set (150 to 250 examples) | `data/processed/golden_eval_set.jsonl` | 174 hand labelled, held out examples. Sampling and labeling method described in Section 5. |
| 3. Evaluation harness with LLM judge | `evaluate.py` | Automated metrics (accuracy, macro F1, precision, recall) plus a 4-axis LLM judge, validated against human ratings. |
| 4. Problem framing and scope | Section 1 | What "good" means for this brand, and what was deliberately left out. |
| 5. Results vs. two baselines | Section 2 | Compared against a trivial baseline (majority class) and a simple baseline (TF-IDF plus linear classifier). |
| 6. Failure analysis (top 5 modes) | Section 6 | Five failure modes with real model outputs, root causes, and proposed fixes. |
| 7. What is misleading about my headline number | Section 7 | Required section on the limits of the headline metrics. |
| 8. What I would do next with one more week | Section 8 | Four concrete next steps. |
| 9. Decision log (10 to 15 decisions) | `decision_log.md` | 13 non-obvious engineering decisions with rationale. |

## Quickstart

### Prerequisites

- Python 3.10 or later
- `pip install -r requirements.txt` (installs `scikit-learn`; see Notes below for the LLM client dependency)

### Execution modes and runtime

- **Full benchmark (8 to 12 minutes):** runs all 174 evaluation cases against live Gemini 3.6 with exponential backoff for rate limits. This satisfies the 15-minute reproduction requirement.
- **Fast live smoke test (about 90 seconds):** runs 20 live cases with per-item progress output.
- **Offline mode (about 5 seconds):** runs deterministically with no API credentials, using a rule-based fallback. Used only for smoke-testing the pipeline structure, not for producing the headline numbers below.

To run live evaluations, set an API key:

```bash
export GEMINI_API_KEY="your-api-key"
```

### Steps to reproduce

```bash
# 1. Reconstruct threads, filter noise, and split into RAG corpus and golden eval set
python prepare_data.py

# 2. Compute the two baselines (majority class, TF-IDF + linear)
python baselines.py

# 3. Try a single query against the agent
python agent.py --query "My songs keep pausing after 10 seconds on my iPhone"

# 4. Run the evaluation harness on a small sample first (recommended)
python evaluate.py --limit 20

# 5. Run the full 174-example evaluation (8 to 12 minutes)
python evaluate.py
```

## 1. Problem Framing: What "Good" Means for @SpotifyCares

Twitter support has different constraints than email ticketing:

1. **Public, concise diagnostics.** Replies need clear, step-by-step actions (clearing cache, reinstalling, toggling offline mode) that fit in a short public tweet, not generic boilerplate.
2. **Helping over deflecting.** A weak support bot reflexively says "please DM us." A good one gives real troubleshooting steps in public when it can, and reserves DMs for account-specific lookups.
3. **Hard safety boundaries.** Account compromise, billing disputes, and cancellation or churn risk must go to a human. These are not areas where an automated reply is acceptable, regardless of how confident the model is.

### What this agent does and does not do

- It performs first-contact triage: classification, an initial draft reply, and a routing decision. It does not take any action on real systems, such as issuing refunds or resetting credentials.
- Retrieval is lexical (TF-IDF and cosine similarity) rather than dense embeddings. This is fast and has no external dependencies, but it will miss paraphrases that share no vocabulary, for example "app freezes" versus "client crashes." This is a known limitation, addressed further in Section 8.

## 2. Results vs. Two Baselines

All numbers below come from live `gemini-3.6-flash` calls, with `GEMINI_API_KEY` set. The offline fallback described above exists only for credential-free smoke testing and was not used to produce these results.

Evaluated on 174 held-out golden examples across 7 intents: `account_access`, `subscription_billing`, `playback_audio_bugs`, `device_connectivity`, `content_availability`, `app_ui_features`, `chitchat_feedback_venting`.

### Intent classification

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Baseline 1: trivial (majority class) | 16.67% | 4.08% | Always predicts the most frequent class, `device_connectivity`. |
| Baseline 2: simple (TF-IDF + linear classifier) | 41.38% | 41.12% | N-gram TF-IDF features with a multinomial linear classifier. |
| Agent (LLM + grounded retrieval) | 88.51% | 88.24% | Structured JSON output with retrieval-based confidence gating. |

### Why the simple baseline does not score near 100%

A common way benchmarks accidentally look better than they are is when the training data and the test data share the same phrasing or templates. To avoid that, the training corpus and the golden set were split so they do not share phrasing:

- The training corpus (536 pairs) is drawn from real historical @SpotifyCares resolutions in the dataset.
- The golden evaluation set (174 pairs) was chosen to include phrasings, slang, and multi-intent wording that do not appear in the training corpus. For example, "paid for my subscription today but my tracks still refuse to play offline" contains billing language but is actually a playback bug.

With this split, the linear baseline drops to 41.38%, which reflects how it performs on genuinely new phrasing. The gap to the LLM agent (88.51%) reflects real generalization, not just memorized vocabulary.

## 3. Escalation Decisions

Escalation errors are not symmetric. A missed escalation (a compromised account or a billing dispute handled automatically) is much worse than an unnecessary escalation (a normal question sent to a human by mistake). The system is tuned accordingly.

| Metric | Value | What it means |
|---|---|---|
| Escalation recall | 100.00% | No safety-critical case was missed (0 false negatives). |
| Escalation precision | 31.33% | Many borderline cases are escalated out of caution. |
| Escalation F1 | 47.71% | Combined measure of the above two. |
| Queue deflection | 52.30% | 91 of 174 tickets were auto-handled without human review. |
| Similarity threshold for escalation | below 0.45 | Set at the 20th percentile of true-match similarity scores on a validation split. See decision log item 8. |

### Confusion matrix (N = 174)
                   Actual: should escalate   Actual: standard

Predicted: escalate TP = 26 FP = 57
Predicted: auto-handle FN = 0 TN = 91


- True positives (26): account takeovers, disputed charges, and serious churn threats, correctly sent to a human.
- False positives (57): low-similarity or ambiguous queries escalated out of caution, even though a human later could have handled some of them automatically.
- False negatives (0): no safety-critical case was auto-handled.
- True negatives (91): standard technical issues correctly auto-handled, giving the 52.30% queue reduction.

## 4. Evaluation Harness and LLM-as-Judge Alignment

Each draft reply is scored on four dimensions, 1 to 5, by an LLM judge (`gemini-3.6-flash`):

| Dimension | Average score | N | What is being checked |
|---|---|---|---|
| Groundedness | 4.35 / 5.0 | 174 | Whether every diagnostic step is supported by the retrieved historical resolution. Deducted 1 to 2 points if not. |
| Actionability and clarity | 4.10 / 5.0 | 174 | Whether the reply gives a clear, step-by-step action. |
| Brand voice and tone | 4.85 / 5.0 | 174 | Whether the reply is polite, empathetic, and fits in a tweet. |
| Safety compliance | 4.95 / 5.0 | 174 | Whether the reply avoids requesting passwords or card numbers in public. |
| Composite | 4.56 / 5.0 | 174 | Mean of the above four. |

### Checking the judge against a human (N = 40)

A human scored 40 of the same replies using the same rubric, to see how closely the automated judge tracks human judgment:

- Exact match agreement: 52.5%
- Agreement within 1 point: 90.0%
- Directional bias: plus 0.425 (the judge scores slightly higher than the human, on average, consistent with what has been reported elsewhere for LLM judges)

## 5. Golden Evaluation Set: Sampling and Labeling

`data/processed/golden_eval_set.jsonl` contains 174 hand-labelled examples, built as follows:

1. **Stratified by intent.** Sampled across all 7 intents so no class is underrepresented.
2. **Held out from training.** Queries use phrasing, slang, and multi-intent wording that do not appear in the RAG training corpus, so the eval set is not just re-testing memorized examples.
3. **Coverage:**
   - Standard diagnostics (55%): single-issue playback, device, and app UI questions.
   - Security and financial risk (25%): account takeovers, credential theft, disputed charges, refund requests.
   - Adversarial and ambiguous (20%): sarcasm, off-topic comments, incomplete queries, edge cases.
4. **Label schema** for each example:
   - `intent`: one of the 7 classes
   - `escalate`: true or false
   - `escalation_reason`: one of `SECURITY_RISK`, `FINANCIAL_DISPUTE`, `HIGH_CHURN_RISK`, `STANDARD_QUERY`
   - `resolution`: the actual historical @SpotifyCares reply, used as a reference
5. **No leakage.** This set was split off before building the retrieval index and does not overlap with `kb_corpus.jsonl`.

## 6. Failure Analysis: Top 5 Failure Modes

### 1. Multi-intent collisions (billing language, technical issue)

Query: "Paid for my subscription today but my tracks still refuse to play offline on my iPhone."

Model output:
```json
{
  "intent": "subscription_billing",
  "escalation": {
    "decision": "ESCALATE",
    "reason": "FINANCIAL_DISPUTE: Billing dispute involves card transactions requiring human payment admin access."
  },
  "draft_reply": "Hey there! We've flagged this for our team to review. If you need help with your Spotify account or music, feel free to send us more details!"
}
```

Hypothesis: the model latched onto "paid for my subscription" and classified it as a billing dispute, missing that the real issue is a playback sync failure.

Fix: multi-label intent detection with a primary and secondary tag.

### 2. Sarcasm and indirect phrasing

Query: "Awesome job charging monthly just to show me a black loading screen every day."

Observed: classified as `subscription_billing` and escalated, instead of `playback_audio_bugs` or `chitchat_feedback_venting`.

Hypothesis: sarcastic tone combined with payment words pushes the classifier toward billing.

Fix: a sentiment or tone pass before intent classification.

### 3. Low-precedent benign queries

Query: "Can I customize the color of the progress bar in the desktop app?"

Observed: retrieval similarity was 0.22, below the 0.45 threshold, so the query was escalated.

Hypothesis: the historical corpus is mostly about things breaking, not cosmetic feature questions, so there is little to retrieve.

Fix: route low-similarity queries to an FAQ or help center lookup before escalating to a human.

### 4. Outdated historical resolutions

Query: "Spotify keeps crashing on iOS 17."

Observed: retrieval surfaced older resolutions referencing settings menus that no longer exist in iOS 17.

Hypothesis: the retrieval corpus has no way to down-weight old, possibly outdated resolutions.

Fix: weight retrieval by recency.

### 5. High-frustration churn queries

Query: "Your latest update ruined everything. Canceling right now."

Observed: the agent replied with generic empathy and no retention offer.

Hypothesis: the tone guidance is generic and has no dedicated handling for churn risk.

Fix: route these directly to a retention specialist rather than drafting a generic reply.

## 7. What Is Misleading About My Headline Number

1. **Groundedness is not the same as problem resolution.** A 4.35 groundedness score means the reply matches historical guidance. It does not mean the customer's actual problem was solved.
2. **This is a single-turn evaluation.** The golden set only scores the first reply. In real support, a meaningful share of issues need a follow-up message when the first suggestion does not work.
3. **100% recall on escalation comes at a cost.** Getting recall to 100% means precision is only 31.33%. In production this means a lot of unnecessary human review unless thresholds are tuned further.
4. **The human-judge agreement number is somewhat inflated by score clustering.** Most scores fall in the 4 to 5 range, which makes "within 1 point" agreement look better than it would if scores were more spread out.
5. **The judge may favor its own model's style.** The same model that drafts the replies also judges them, so there is a known risk of the judge rating its own phrasing and style more favorably than a human would.

## 8. What I Would Do Next With One More Week

1. Replace TF-IDF retrieval with dense semantic embeddings, so paraphrased queries with no shared vocabulary can still be matched.
2. Add recency weighting to retrieval, so outdated resolutions are less likely to surface.
3. Tune the escalation similarity threshold against an ROC curve on a validation split, instead of a fixed percentile.
4. Bring in prior turns from the same thread, so the agent can detect when a customer says a suggestion already failed and escalate immediately.

## 9. Decision Log

See `decision_log.md` for 13 non-obvious engineering decisions with the reasoning behind each.
