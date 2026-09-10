# Decision Log

This log covers the non-obvious engineering decisions made while building the
@SpotifyCares support agent, and the reasoning behind each one.

## 1. Brand selection: @SpotifyCares over Amazon or Delta

Decision: chose Spotify support over higher-volume brands like Amazon
(@AmazonHelp) or Delta (@Delta).

Why: over 85% of Amazon and Delta tweets in the dataset are dead-end
deflections, for example "please DM us your order ID." Spotify's support
account routinely gives real diagnostic steps in public tweets (clearing
cache, toggling offline mode, reinstalling, checking family plan settings).
That gives the retrieval step something genuinely useful to ground on.

## 2. Regex plus keyword exception for filtering non-resolutions

Decision: instead of filtering by reply length alone, used a two-part rule.
Drop any reply matching a pure DM deflection pattern
(`\b(send (us )?a dm|dm us|check inbox)\b`), unless the same reply also
contains a concrete troubleshooting keyword (`cache`, `reinstall`, `restart`,
`toggle`, `settings`, `http`).

Why: a length-only filter either throws away useful replies that end with an
optional DM offer, or lets through thousands of generic "DM us" replies that
would pollute retrieval.

## 3. In-memory vector search instead of a vector database

Decision: built an in-memory TF-IDF and cosine similarity retriever in plain
Python, instead of using ChromaDB, FAISS, or Pinecone.

Why: for a corpus of 536 historical resolutions, an in-memory search runs in
under 15ms on CPU, with no external services, no extra build dependencies,
and no network calls. That keeps the pipeline simple to run and reason about.

## 4. No external API calls for retrieval

Decision: LLM API calls are used only for intent classification, reply
drafting, and judging. Retrieval runs entirely on the local lexical index,
with no embedding API calls.

Why: free-tier API quotas are limited. Spending calls on embeddings for
hundreds of historical pairs risks hitting rate limits during evaluation for
no real benefit at this corpus size.

## 5. Exponential backoff with jitter on all LLM calls

Decision: wrapped every LLM API call in an exponential backoff loop with
random jitter (`time.sleep(2**attempt + uniform(0.5, 1.5))`).

Why: prevents the evaluation run from failing partway through the 174
examples when the API throttles concurrent requests.

## 6. Strict separation between the training corpus and the golden set

Decision: the 174 golden evaluation examples were split off from the 536-pair
training corpus using phrasing, slang, and multi-intent wording that does not
appear in the training data.

Why: if test queries or similar phrasing show up in the retrieval corpus, the
retrieval and reply quality numbers get inflated by the model effectively
looking up the answer rather than generalizing.

## 7. Escalation tuned for recall over precision

Decision: tuned the escalation logic for 100% recall on critical cases
(account theft, billing fraud, high churn risk), accepting lower precision as
a result. Final numbers: 31.33% precision, confusion matrix TP 26, FP 57,
FN 0, TN 91.

Why: in support, a missed escalation (a hacked account or a billing dispute
handled automatically) is a much worse outcome than an unnecessary escalation
(a normal question sent to a human by mistake). The cost is asymmetric, so
the system is tuned accordingly.

## 8. Escalation threshold set from the data, not guessed

Decision: queries with retrieval similarity below 0.45 are escalated. This
threshold is the 20th percentile of true-match similarity scores on a
validation split.

Why: when there is no good historical match for a query, letting the model
draft a reply anyway tends to produce generic or made-up answers. A threshold
derived from actual similarity scores, rather than a guessed number, gives a
defensible cutoff for when to escalate instead.

## 9. Two separate baselines

Decision: compared the agent against a trivial baseline (majority class:
16.67% accuracy, 4.08% macro F1) and a simple baseline (TF-IDF plus linear
classifier: 41.38% accuracy, 41.12% macro F1).

Why: this shows the intent taxonomy is not trivially solvable, and gives a
real basis for judging how much the LLM agent (88.51% accuracy) is actually
adding on unfamiliar phrasing, rather than just beating an easy strawman.

## 10. Four separate judge dimensions instead of one quality score

Decision: the LLM judge scores each reply on four separate 1 to 5 scales:
groundedness, actionability, brand voice and tone, and safety.

Why: a reply can be well-written and on-brand while still making up details
that are not in the retrieved historical resolution. A single quality score
would hide that. Separate scores make specific failure types visible.

## 11. Reporting judge agreement as exact match and within-1-point

Decision: measured human versus judge agreement as exact match percentage
(52.5%) and agreement within 1 point (90.0%), along with the average
directional bias (plus 0.425), rather than only a correlation coefficient.

Why: correlation can look strong even when scores cluster tightly together
and rarely match exactly. Reporting direct agreement percentages is easier to
read and interpret at face value.

## 12. No multi-turn state machine, no live actions on real systems

Decision: scoped the agent to first-contact triage, drafting, and routing
only. It does not take real actions such as issuing a refund or resetting a
password.

Why: giving an automated system write access to billing or account systems
without a human in the loop is a real compliance and security risk. That
was out of scope for this project regardless of how well classification and
drafting performed.

## 13. Offline fallback mode

Decision: included a deterministic, dependency-free fallback for
classification, retrieval, and evaluation that runs without any API access.

Why: lets someone check that the full pipeline runs end to end even without
an API key, as a basic smoke test. It was not used to produce any of the
headline numbers in the README.