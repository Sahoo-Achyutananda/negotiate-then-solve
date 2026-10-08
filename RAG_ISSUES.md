# Policy check and RAG: known issues and planned fixes

Working notes for the `f-rag` branch: how the policy check works today, the
request messages that are known to fail, and the fixes planned for retrieval
and parsing.

## How the policy check works

The policy check runs after a message is parsed and before anything reaches
the solver. It decides whether a request is **allowed**, **needs approval** or
is **forbidden** under the institute's rules, and it answers policy
questions. Who may ask for what (authority) is checked earlier, by the code
that post-processes the parse.

```
message ──► parse ──► query = text + hints ──► retrieve top 5 rules ──► verdict ──► checks ──► act
```

1. **Rules (at start-up).** `load_corpus()` in `src/agents/documents.py` reads
   `data/handbook.md` (one rule per `## <number> <title> [RULE-ID]` heading,
   16 rules) and every document in `data/policies/` (text PDF, scanned PDF,
   photo, web page; OCR where needed), cut into rules at numbered headings
   (16 more). 32 rules in all, each with its source document, page and method.
2. **Which requests are checked** (`Orchestrator.submit`,
   `src/pipeline/orchestrator.py`): `answer` (policy questions) and `compile`
   (requests to change something); swaps and extra classes have their own
   calls. `clarify`, `refuse`, `investigate` and `out_of_scope` exit first.
3. **Query.** The message text plus `query_hints()` (`src/agents/policy.py`):
   words the parse implies but the message may not say ("absence missed class
   make-up" for an absence in given weeks, "lunch break" for the lunch slot,
   "consecutive hours in a row" for a long run, "swap slots", ...).
4. **Retrieval.** Top 5 rules from the hybrid retriever (the web app's
   default, `NTS_RETRIEVER`): BM25 (words, with clock times normalised) and
   dense embeddings (`BAAI/bge-small-en-v1.5` through fastembed), merged by
   reciprocal rank fusion with equal weight.
5. **Verdict**, one of two ways:
   - **With a model** (`--parser local` / `gemini`), `PolicyAgent`: the model
     gets the 5 rules, the calendar, the parser's output and the message, and
     returns `verdict`, `cited_rules`, `obligations`, `explanation`,
     `alternative`, `answer` as JSON. `check()` then drops citations of rules
     that were not shown (`hallucinated`) and turns a denial without a valid
     citation into `needs_approval`. If the model does not answer, the
     `Fallback` wrapper uses the rule-based agent for that request.
   - **Offline / model down**, `RulePolicyAgent`: a question is answered with
     the text of the **top-1** rule; a hard request for the lunch slot is
     forbidden (P-LUNCH); a run of more than three hours, when the text says
     "back to back" or similar, is forbidden (P-MAXCONSEC); an absence in
     given weeks is allowed with a P-MAKEUP duty; everything else is allowed.
6. **Acting on it.** `forbidden` → DENIED with the cited rule and the
   alternative; `needs_approval` → ESCALATED to the coordinator; `allowed` →
   compiled and solved, with any duties recorded; a question → ANSWERED.

Every step is shown on the request's page in the web app (retrieved and cited
rules, verdict, hallucinated citations).

### Weaknesses of the current design

- Offline, only **2 of the 32 rules** can deny anything; the rest are
  retrieved but never applied (a Saturday class is allowed, although P-HOURS
  forbids weekend classes without approval).
- The check depends on the parse: if the parser drops a detail ("Saturday"),
  neither the hints nor the rule-based checks can see it.
- Retrieval always returns 5 rules: there is no "no relevant rule" outcome, so
  questions can be answered from a loosely related rule.
- `needs_approval` sends the request to the coordinator; whose approval a rule
  actually requires (the Dean for the lunch hour, the head of department for
  out-of-hours classes) is not used.

## Retrieval today

`uv run python -m evaluation.retrieval` (no API calls; 32 rules, test split;
report in `runs/retrieval-20261009-011442.json`). Recall@k is the share of
gold rules in the top k.

| Query set | Retriever | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|
| Corpus requests, raw text (106) | BM25 | 0.387 | 0.604 | 0.774 | 0.513 |
| | Dense | 0.151 | 0.292 | 0.453 | 0.243 |
| | Hybrid | 0.236 | 0.585 | 0.755 | 0.433 |
| Corpus requests + parse hints (106) | BM25 | 0.783 | 0.811 | 0.877 | 0.810 |
| | Dense | 0.802 | 0.868 | 0.915 | 0.841 |
| | Hybrid | 0.774 | 0.877 | 0.934 | 0.836 |
| Paraphrased questions (40) | BM25 | 0.415 | 0.561 | 0.658 | 0.515 |
| | Dense | 0.683 | 0.780 | 0.878 | 0.771 |
| | Hybrid | 0.634 | 0.780 | 0.829 | 0.721 |

Observations:

1. **Top-1 is weak.** The offline agent answers questions from the top-1 rule
   alone, so about one answer in three quotes the wrong rule.
2. **Vocabulary mismatch.** Misses from the paraphrase set: "four periods
   without stopping" does not reach P-MAXCONSEC, "the lecture I lose while
   I'm at a workshop abroad" does not reach P-MAKEUP, "trade our Tuesday and
   Thursday lectures" does not reach P-SWAP.
3. **Hybrid can lose to BM25 alone** (raw text R@1 0.236 vs 0.387): equal-weight
   fusion lets the weaker dense ranking pull it down.
4. **No abstention**: dense always returns k rules.
5. **Small evaluation set**: 40 paraphrased queries, written by one author.

## Requests known to fail

Run through System One, the rule parser and the rule-based policy agent
(offline mode), as Dr. Sharma (F-107, Machine Learning lectures, Algorithms
lectures and practical) on the demo department. **All but one took System
One's fast path**, so they are read by the rule parser even when a model is
connected: connecting the LLM does not fix them.

### Severe: a wrong constraint that blocks or forces too much

| # | Message | Read as | Should be |
|---|---|---|---|
| 1 | I can't teach on Thurdsay in week 9 | Unavailable for all of week 9 (the misspelt day is dropped) | Thursday of week 9, or a question |
| 2 | I'm not available in the evenings in week 8 | Unavailable for all of week 8 | A question: there are no evening slots (teaching ends at 17:00) |
| 3 | Shift my ML lecture to Monday at 11 am | Every class she teaches must be on Mon 11 am (hard) | The Machine Learning lecture ("ML" is not recognised) |
| 4 | Please move my Compiler Design lecture to Friday at 10 am | Every class she teaches must be on Fri 10 am (hard) | Refused or a question: she does not teach Compiler Design |
| 5 | I have a medical appointment next Tuesday afternoon | Unavailable every Tuesday afternoon, all semester | One Tuesday, with a make-up owed ("next Tuesday" is ignored) |
| 6 | Can I take my machine learning lecture on Saturday morning in week 6? | "Would rather not have classes 9 am–1 pm" in week 6 | Needs approval or forbidden under P-HOURS; Saturday is dropped and the meaning inverted |

### Times and dates misread

| # | Message | Read as | Should be |
|---|---|---|---|
| 7 | Move my machine learning lecture to 14:00 on Friday | Friday, any time | Friday 2 pm (24-hour times are not read) |
| 8 | Please schedule my machine learning lecture at noon on Wednesday | Wednesday, any time | Wednesday 12 pm ("noon" is not read) |
| 9 | Move my machine learning lecture to Friday after 3 | Friday, any time | Friday from 3 pm ("after 3" without "pm") |
| 10 | Keep my Monday classes between 10 am and 12 pm | Avoid 10 am and 12 pm | Prefer 10 am–12 pm (range and "keep" lost) |
| 11 | I'm away for weeks 5 through 7 for a conference | Week 5 only | Weeks 5–7 ("through" is not a range) |
| 12 | I can't teach on Monday in week 3 or on Friday in week 4 | Mon and Fri, in week 3 | Mon of week 3 and Fri of week 4 |

### Wrong strength or wrong class

| # | Message | Read as | Should be |
|---|---|---|---|
| 13 | I cant take classes on Monday in week 5 | Soft preference (tier 5) | Unavailability (tier 3); "cant" without the apostrophe is not recognised |
| 14 | My algorithms practical needs a bigger lab with GPUs, preferably on Wednesday | Wednesday required (hard) | Wednesday as a wish (soft): "preferably" is ignored |
| 15 | Move my machine learning lecture to Tuesday and my algorithms practical to Thursday | Only the practical, on Tue or Thu | Two moves; the ML lecture is lost |
| 16 | I'd like my machine learning lectures on Monday, Wednesday and Friday mornings | All her classes | Only the ML lectures |

### Questions not recognised as questions

All sent back with "Could you tell me which days and times you mean?".

| # | Message | Should be |
|---|---|---|
| 17 | Tell me the rule about swapping lectures | Answered from P-SWAP (no "?", so not seen as a question) |
| 18 | Is it okay to teach four periods without stopping? | Answered from P-MAXCONSEC |
| 19 | What happens to the lecture I lose while I'm at a workshop abroad? | Answered from P-MAKEUP |
| 20 | Can I hold a quiz in my machine learning lecture in week 8? | Forwarded to the exam cell, or out of scope |

## Issues to open

Grouped by cause:

1. **Rule parser: a misread day or time can block a whole week** (#1, #2, #6).
   Highest priority.
2. **Rule parser: an unknown course name applies the constraint to all of a
   teacher's classes** (#3, #4, #16).
3. **Rule parser: time formats**: 24-hour, "noon", bare "after 3", ranges
   (#7–#10).
4. **Rule parser: week ranges, relative dates and several requests in one
   message** (#5, #11, #12, #15).
5. **Rule parser: wording that changes strength**: "cant", "preferably",
   "keep" (#13, #14, #10).
6. **System One: questions without "?" or policy words are sent back for
   clarification** (#17–#20).

A cross-cutting option for 1–5: keep the fast path off whenever the rule
parser's result looks risky (a hard constraint on all of someone's classes, a
whole week blocked, a course the sender does not teach), so the model reads
those messages when one is connected.

## Planned RAG fixes

Each is one change, measured with `evaluation.retrieval` before and after.
1–3 are the core.

| # | Fix | Addresses | Notes |
|---|---|---|---|
| 1 | **Cross-encoder reranker**: take the hybrid top 10 and re-score each (query, rule) pair with `BAAI/bge-reranker-base` or `Xenova/ms-marco-MiniLM-L-6-v2` (both supported by the installed fastembed 0.8.1 as `TextCrossEncoder`); keep the top 5 | Top-1 accuracy, vocabulary mismatch | Offline on CPU, roughly 20–50 ms per query |
| 2 | **Rule enrichment (doc2query)**: index each rule with its document title and section, plus a few questions it answers ("Can I teach four hours straight?"), generated once by a model and cached | Vocabulary mismatch, for BM25 and dense alike | One generation pass, then offline |
| 3 | **Abstention**: a score threshold on the reranker; below it, return no rule, so a question goes to the coordinator instead of being answered from an unrelated rule | Confident answers from the wrong rule | Needs out-of-scope queries in the evaluation set |
| 4 | **Query rewriting from the parse** (HyDE / multi-query with the local model): "4 hours from 9" → "consecutive teaching hours" | Requests whose rule is implied, not stated | Medium effort |
| 5 | **Fusion tuning**: weight BM25 against dense, and choose k, on the validation split | Hybrid losing to BM25 | Small |
| 6 | **Bigger evaluation set**: more paraphrased queries, look-alikes that should retrieve a different rule, out-of-scope questions labelled "no rule"; per-rule report | Trustworthy numbers | Ideally with the team's hand-written requests |
| 7 | **End-to-end check**: `evaluation.policy --local` before and after, to see whether better retrieval changes verdicts and citations | Shows the effect on decisions | Needs the GPU server |

Out of scope for now: turning every rule into a constraint the code checks
itself, a vector database (a matrix product is enough for 32 rules), and
fine-tuning the embedding model.

### Reproducing

```sh
uv run python -m evaluation.retrieval              # retrieval benchmark, no API calls
uv run python -m evaluation.policy --split val     # parse-then-policy evaluation (model needed)
```

The failing requests above can be replayed in the web app (sign in as
Dr. Sharma) or through `Orchestrator._parse` and `policy.review` on a `World`
built with `parser_mode="offline"`.
