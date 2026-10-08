# Customer Risk Intelligence Platform

Predicts whether a customer is likely to become **high-risk in the
next 30 days**, explains why, and answers grounded questions about a
customer using their structured history, the model output and their
documents (RAG).

```
Raw CSVs (synthetic, with injected quality issues)
   │
   ▼
Data engineering ── profile → validate → clean → quarantine → atomic promote
   │                (full refresh or incremental batch upsert)
   ├──────────────► PostgreSQL  (serving copy of clean data)
   ▼
Feature engineering ── monthly snapshots, 90-day lookback, 30-day target
   │
   ▼
ML ── baseline / Logistic Regression / Random Forest / XGBoost → SHAP
   │
   ▼
FastAPI ─┬─ POST /predict-risk
         ├─ POST /customer-intelligence
         │        ├─ customer master + transactions + interactions
         │        ├─ ML prediction (only if a snapshot exists)
         │        └─ RAG: query plan → Qdrant (customer + type filter)
         │                 → threshold → gpt-4o-mini → validated citations
         ├─ GET  /customers/{id}   (history from PostgreSQL)
         ├─ GET  /health
         └─ /ui  demo dashboard
```

## Quick start

```bash
cp .env.example .env            # set DATABASE_PASSWORD and OPENAI_API_KEY

# Local
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make pipeline                   # data → clean → features → train → SHAP
make db-load ingest             # PostgreSQL + Qdrant
make api                        # http://localhost:8000/ui/  and  /docs

# Docker
docker compose --profile pipeline run --rm pipeline   # first time only
docker compose up --build
```

`make all` runs the full chain, including the RAG evaluation, the
pipeline demos and the tests.

---

## 1. Dataset generation

Synthetic data (Faker + NumPy + Pandas, fixed seeds) because real
customer, payment and support data can't be shared, and generating
it lets me control the risk signal and inject known quality problems
to test the pipeline against.

| Dataset | Raw rows | Clean rows |
|---|---:|---:|
| Customers | 1,005 | 1,000 |
| Transactions | 30,020 | 29,955 |
| Interactions | 15,020 | 14,980 |
| Documents | 305 | 285 |

Data window: 2025-01-01 to 2025-12-31. All processing uses a fixed
as-of date (`DATA_AS_OF_DATE = 2025-12-31`) rather than "today", so
reruns are deterministic.

`customer_since` is generated inside the data window. An earlier
version used Faker's `end_date="today"`. That put 43% of transactions
before the customer's own onboarding date, and the output depended
on the day the generator ran. The fix is in `generate_customers.py`,
and the processed validator now checks `transaction_date >=
customer_since`.

**Injected issues** (`inject_quality_issues.py`): missing values
(country, industry, payment status, sentiment, document content and
source), duplicate records in every dataset, invalid categorical
values, unknown customer IDs, unparseable dates, negative amounts and
999,999,999 amount outliers.

## 2. Data quality and engineering

`profiler.py` → `validator.py` (raw) → `cleaner.py` →
`processed_validator.py` (fails the run with a non-zero exit code if
any check fails).

| Issue | Handling |
|---|---|
| Duplicates | Dropped by primary key |
| Missing industry / country / source | Filled with `Unknown` / `UNKNOWN` |
| Missing payment status | Derived: paid date → PAID; past due and unpaid → OVERDUE; otherwise PENDING |
| Missing sentiment | `NEUTRAL` |
| Invalid categories | Normalized to a safe default (`UNKNOWN`, `PENDING`, …) |
| Unknown customer (orphans) | **Quarantined** (`UNKNOWN_CUSTOMER`) |
| Invalid dates, due date before transaction | Quarantined |
| Amount ≤ 0 | Quarantined |
| Amount > 1,000,000 | Quarantined (`AMOUNT_OUTLIER`). This is a fixed business cap, not an IQR, so an incremental batch is judged by the same rule as the full load. |
| Transaction before `customer_since` | Quarantined |
| Empty document content | Quarantined |

Rejected rows go to `data/quarantine/<dataset>_quarantine.csv` with a
`quarantine_reason`. They are kept for review, never silently dropped.

### Operational behaviour

All five behaviours are reproducible with `make demo`, which runs on a
temp copy of the raw data. Results are in
[docs/pipeline_demo_results.md](docs/pipeline_demo_results.md).

- **Idempotency.** Cleaning is a deterministic full refresh. Running
  it twice gives identical row counts, zero duplicate keys and
  byte-identical files (same SHA-256).
- **Failure halfway.** Every output is first written to
  `data/processed/_staging/<run_id>/` and only moved into place
  (`os.replace`, atomic per file) once all datasets are cleaned. The
  manifest (`_manifest.json`: run id, row counts, hashes) is written
  last. In the demo the run crashes while cleaning interactions. The
  processed files and manifest stay unchanged, staging is cleaned up,
  and the rerun produces identical output with no duplicates.
- **Incremental processing.** `python -m src.data_engineering.incremental <batch_dir>`
  runs: schema check → referential integrity → the same `transform_*`
  functions as the full load → upsert by primary key
  (insert/update/unchanged) → staged promotion. Each batch is recorded
  with a content fingerprint, so re-applying a batch is a no-op.
- **Late-arriving data.** New records dated before the newest existing
  record are accepted and flagged. Corrections are flagged too. The
  batch report says from which date feature snapshots must be rebuilt
  and for which customers.
- **New columns.** Each dataset has a column contract. An unknown
  column is logged as a schema change (`[SCHEMA] … new column(s)
  detected`) and dropped, so unreviewed fields never reach features. A
  missing required column fails the run with `SchemaValidationError`
  and leaves processed data untouched.
- **Deleted or unknown customers.** The raw referential-integrity
  check fails (20 orphan transactions). Cleaning quarantines them and
  0 orphans reach clean data.

`python -m src.database.load_processed` syncs clean data into
PostgreSQL. It upserts by primary key and deletes rows that no longer
exist, all in one transaction, so it is idempotent.

## 3. Machine learning

**Target.** For each customer and month-end observation date, a
business risk score is computed from what happens in the **next 30
days**:

| Signal in the 30-day window | Weight |
|---|---:|
| ≥1 transaction overdue by end of window | 0.30 |
| ≥1 failed payment | 0.25 |
| ≥1 complaint | 0.15 |
| ≥1 negative interaction | 0.10 |
| ≥1 escalation | 0.10 |
| Activity drops below 50% of previous period | 0.10 |

`target = 1` when the score is ≥ 0.50.

**Leakage prevention.** Features only use data on or before the
observation date (90-day lookback, 30-day "recent" window). The target
only uses data after it. Observation dates stop 30 days before the
end of the data so every target window is complete. No snapshot is
created before a customer joined. The target column is excluded from
the feature list. These rules are covered by `tests/test_features.py`.

**Features (24).** Tenure, transaction totals and averages, overdue
count and amount, failed count, failure rate, average payment delay,
frequency, recent vs. previous activity and activity change, pending
count, interaction / complaint / support / negative / escalation
counts (total and recent), and document counts. Segment, industry,
country and account status are one-hot encoded.

**Time-based split.** Train on 2025-04-30 … 2025-09-30 (3,250 rows,
24.7% positive), test on 2025-10-31 … 2025-11-30 (1,776 rows, 38.7%
positive). The positive rate rises in the test period. That drift is
real, and it's one reason PR-AUC and recall matter more than accuracy
here.

**Results on the test period**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (majority class) | 0.000 | 0.000 | 0.000 | 0.500 | 0.387 |
| **Logistic Regression** | 0.593 | **0.616** | **0.604** | **0.723** | 0.611 |
| Random Forest | **0.616** | 0.522 | 0.565 | 0.722 | **0.613** |
| XGBoost | 0.586 | 0.368 | 0.452 | 0.670 | 0.556 |

Confusion matrix for the selected model: TN 797, FP 291, FN 264, TP 424.

**Selection.** Rank by PR-AUC, but treat differences below 0.01 as a
tie, since a 0.002 gap on ~1,800 rows is noise. Break ties on recall,
because missing a customer who becomes high-risk costs more than an
extra review. Random Forest and Logistic Regression tie on PR-AUC.
Logistic Regression has clearly better recall (+9.4 pts) and F1, and
it is simpler and cheaper to serve, so it is selected. The rule and
the shortlist are saved in `models/model_selection.json`.

**Explainability.** SHAP is computed for the selected model. The API
turns the top positive SHAP contributors into business wording
("4 failed payments", "Overdue amount of 599,002.48"). Globally,
`customer_tenure_days` dominates (mean |SHAP| 0.66, next is recent
transaction count at 0.19). I'd treat that as a property of the
synthetic generator (newer customers have short histories, which
trips the activity-decline signal) rather than a real-world finding.
It would be the first thing to check on real data.

Risk levels: HIGH ≥ 0.60, MEDIUM ≥ 0.30, LOW otherwise.

## 4. RAG / customer intelligence

**Ingestion** (`ingest_documents.py`). Clean documents are split with
`RecursiveCharacterTextSplitter` (800 chars, 120 overlap) and embedded
with `text-embedding-3-small`. They are stored in Qdrant (cosine) with
metadata `customer_id`, `document_id`, `document_type`,
`document_date`, `source` and `content_hash`, plus keyword payload
indexes on the filter fields. Point IDs are deterministic
(`uuid5(document_id:chunk)`). Reruns only embed new or changed
documents and delete removed ones. `--rebuild` recreates the
collection.

**Retrieval** (`retriever.py`). Every question first gets a query
plan:

| Plan | When | Behaviour |
|---|---|---|
| `typed` | Mentions a supported document type (payment, complaint, renewal, …) | Customer filter + `document_type` filter (`MatchAny`), query expanded with type vocabulary |
| `general` | Broad questions (risk, evidence, history, summary) | Customer filter, all types, risk vocabulary expansion |
| `unsupported` | Topics we hold no data for (legal, lawsuit, credit score, revenue, …) | **No search** |
| `none` | No document topic | No search; structured data only |

Results below a cosine similarity of 0.35 are dropped, and the
customer ID is checked again on every hit. I did not lower the
threshold. Short questions such as "What customer preference was
recorded?" scored poorly because they share few words with the
notes. Query expansion fixed that: typed queries now score 0.50–0.71,
and the broad risk question went from 0.30 to 0.42.

**Generation** (`customer_intelligence.py`):

1. **Customer existence comes from the customer master**, not the ML
   feature table. 76 customers joined after the last observation
   date. They are valid customers with history and documents, but no
   ML snapshot. They get "ML risk prediction is unavailable because
   this customer does not have sufficient historical data for the
   current prediction framework", never an invented probability.
2. Structured context comes from independent sources: profile,
   transaction summary and the 10 most recent transactions,
   interaction summary and the 10 most recent interactions, plus the
   model prediction only when one exists.
3. `gpt-4o-mini`, temperature 0, JSON output and strict grounding
   rules. Documents are wrapped in `<document>` tags and declared as
   data, not instructions.
4. Post-checks:
   - Citations are kept only if the ID was actually retrieved for
     this customer, and their metadata comes from our retrieval, not
     the model.
   - An answer that mentions any other customer ID is replaced with
     the fallback.
   - Insufficient evidence returns exactly *"I could not find
     sufficient information in the available customer data."* with no
     sources and `grounded=false`.

## 5. Evaluation

`make eval` builds `data/evaluation/rag_evaluation.json` and runs it.

- **10 document questions**: all 7 document types plus 3 rephrased
  questions. Each targets a customer with exactly one document of
  that type, so the expected source is unambiguous. Half are
  customers without an ML snapshot (the group that used to get
  "Customer not found").
- **4 structured questions**: failed vs. overdue counts (the customer
  is chosen so the two numbers differ), profile facts, model
  risk level and probability, and the same risk question for a
  customer with no snapshot.
- **3 unavailable questions**, verified against the data before
  saving:
  - a customer with documents but no preference document (there is
    no preference field in structured data)
  - a customer with no renewal document **and** no RENEWAL
    interaction
  - a legal-dispute question (unsupported topic, no document mentions
    one)

  The retriever must return nothing for each. Cases are **not**
  filtered on the assistant's answer, because that is what is being
  measured.

Metrics apply per case type. N/A is excluded rather than counted as
a failure.

| | Retrieval | Answer | Citation | Grounding |
|---|:-:|:-:|:-:|:-:|
| document | expected doc in retrieved set | ≥60% of the document's key facts, no contradiction | cited docs belong to the customer, include the expected doc and support the answer | flag true, every number in the answer appears in the evidence, ≥70% of content words supported |
| structured | N/A | every expected fact or number present (counts must sit next to their label) | N/A | as above |
| unavailable | nothing retrieved | exact fallback | no sources | `grounded=false`, no sources |

Thresholds were fixed before running and not tuned afterwards.

**Results** (17 questions, final run):

| Metric | Result |
|---|---:|
| Retrieval accuracy | **100.0%** (13/13) |
| Answer correctness | **94.1%** (16/17) |
| Citation correctness | **100.0%** (13/13) |
| Grounding | **94.1%** (16/17) |
| Hallucination rate on unavailable questions | **0%** (0/3) |
| Average latency | 1,428 ms |
| Median latency | 1,517 ms |
| P95 latency | 1,896 ms |

By type: document questions 100 / 90 / 100 / 100 (retrieval / answer /
citation / grounding), structured 100 / 75 (answer / grounding), and
unavailable 100 on every check. Full per-case output is in
`data/evaluation/rag_evaluation_results.json`, with the summary in
`rag_evaluation_summary.csv`.

Remaining failures, left as they are:

- **RAG_003** (support note): the answer mentions the account-related
  issue but leaves out that guidance was provided, so it covers 1 of 2
  key facts. It's a partial answer and is counted as one.
- **RAG_014** (risk question, customer without a snapshot): the answer
  is correct ("ML risk prediction is unavailable…", no invented
  probability), but the model labels its own answer `grounded=false`,
  so grounding fails. Adding a rule that overrides the flag for this
  one sentence would just be fitting the metric. I'd rather move
  grounding to a deterministic server-side check (see limitations).

## 6. Reliability

| Failure | Behaviour | Test |
|---|---|---|
| OpenAI timeout or error | 20 s timeout, 1 retry, then `503` with "temporarily unavailable". The insufficient-information fallback is **not** used here, because it would wrongly tell the user the data doesn't exist. | `test_llm_unavailable` |
| Invalid LLM output (not JSON, wrong types) | `502`, no partial answer | `test_invalid_llm_response` |
| Empty retrieval | Exact fallback, no sources | `test_empty_retrieval_fallback` |
| Wrong customer | Qdrant filter + second check on each hit + citation whitelist + answer scan for other customer IDs | `test_retriever_filters_by_customer`, `test_citations_must_come_from_retrieved_documents` |
| Unsupported topic | No retrieval, no LLM call, fallback | `test_unsupported_topic_*` |
| Prompt injection (question or document) | Only this customer's data is ever in the prompt, documents are delimited as data, and answers naming other customers are suppressed | `test_prompt_injection_*` |
| Qdrant down mid-request | Degrades to structured data and reports `error: Document search unavailable` | `test_vector_store_down_degrades_to_structured` |
| Qdrant / OpenAI key missing at startup | `/customer-intelligence` returns `503`, and the next request retries | `test_customer_intelligence_service_cannot_start` |

**Cost controls**

- Top 4 documents per question, each capped at 1,500 characters.
- Output capped at 400 tokens.
- Structured context is summarized (counts, plus the 10 most recent
  transactions and interactions) instead of the full history.
- No retrieval and no LLM call for unknown customers, unsupported
  topics or invalid requests. No embedding call for questions without
  a document topic.
- Questions are capped at 500 characters.
- `gpt-4o-mini` for generation and `text-embedding-3-small` for
  embeddings.
- Embeddings are only regenerated for new or changed documents
  (content hash). A rerun on unchanged data makes 0 embedding calls.
- Data and models are loaded once at startup, and SHAP background is
  built once.

## 7. API

Interactive docs: `http://localhost:8000/docs`. Demo UI:
`http://localhost:8000/ui/`.

**`GET /health`**: per-component status (`model`, `database`,
`vector_store`, `llm`). Overall status is `ok` or `degraded`.

**`POST /predict-risk`**

```json
{"customer_id": "CUST_00004"}
```
```json
{
  "customer_id": "CUST_00004",
  "observation_date": "2025-11-30",
  "risk_probability": 0.496805,
  "risk_level": "MEDIUM",
  "risk_factors": ["Overdue amount of 599,002.48", "2 pending transactions",
                   "4 failed payments", "Average payment delay of 5.0 days"],
  "model_name": "logistic_regression",
  "model_version": "v1"
}
```

**`POST /customer-intelligence`**

```json
{"customer_id": "CUST_00123", "question": "What payment problems has this customer experienced?"}
```
```json
{
  "customer_id": "CUST_00123",
  "question": "What payment problems has this customer experienced?",
  "answer": "This customer has experienced 5 failed transactions and 1 overdue transaction, with a total overdue amount of 154050.57.",
  "status": "answered",
  "grounded": true,
  "sources": [],
  "error": null
}
```

**`GET /customers/{customer_id}`**: profile, payment status counts,
recent transactions and interactions, and documents (from
PostgreSQL).

**Errors** always use a JSON `detail` body and never include stack
traces:

| Case | Status |
|---|---|
| Missing, empty or malformed `customer_id`, empty or too-long question, invalid JSON | 422 |
| Customer does not exist | 404 |
| Customer exists but has no ML snapshot (`/predict-risk`) | 422 with the "unavailable" message |
| Model not loaded | 503 |
| Database unavailable | 503 |
| Qdrant or OpenAI unavailable | 503 |
| Invalid LLM response | 502 |

## 8. Docker

```bash
cp .env.example .env
docker compose --profile pipeline run --rm pipeline   # builds ./data and ./models
docker compose up --build
```

Services:

- `postgres`
- `qdrant`
- `init`: one-shot job that loads PostgreSQL and syncs embeddings
- `api`: starts after `init` succeeds

Secrets and credentials come from `.env` only. Nothing is hard-coded,
and `.env` is git- and docker-ignored. Host ports are configurable
(`API_PORT`, `POSTGRES_HOST_PORT`, `QDRANT_HOST_PORT`) in case 5432 or
6333 are already in use.

## 9. Testing

```bash
pytest          # or: make test
```

| File | Covers |
|---|---|
| `test_database.py` | `SELECT 1`, data loaded (skipped if PostgreSQL isn't reachable) |
| `test_data_validation.py` | No duplicate IDs, no orphans, valid categories, no required nulls, no outliers, no activity before onboarding, minimum sizes |
| `test_data_cleaning.py` | Each cleaning rule and quarantine reason, schema drift, idempotency, failure recovery, incremental upsert and re-apply |
| `test_features.py` | Leakage checks, complete target windows, time-based split |
| `test_model.py` | Model loads, probabilities in [0, 1], beats baseline |
| `test_api.py` | All endpoints and every error status |
| `test_rag.py` | Query planning, customer and type filtering (in-memory Qdrant), threshold, unsupported topics, fallback, LLM failures, invalid output, citations, prompt injection, prompt size limits |

The RAG and API tests replace OpenAI and Qdrant with fakes, so the
suite runs offline in a few seconds.

## Project layout

```
src/
  data_generation/   synthetic data + quality issue injection
  data_engineering/  profiler, validators, cleaner, incremental, pipeline_demo
  features/          snapshot feature + target builder
  ml/                dataset split, training, SHAP
  database/          SQLAlchemy models, PostgreSQL sync
  rag/               ingestion, retriever, customer intelligence, evaluation
  api/               FastAPI app, routers, validation
frontend/            static demo UI (served at /ui)
tests/
docs/                pipeline demo results
```

## Known limitations / next steps

- The ML context in the assistant uses batch predictions from
  `explain_model.py`, scored on the same snapshot and model as
  `/predict-risk`. A production version would call the same
  predictor directly.
- Late-arriving data is detected and the affected snapshot range is
  reported, but features are rebuilt with a full (fast,
  deterministic) feature run rather than a targeted recompute.
- The `grounded` flag is self-reported by the LLM. The evaluation
  checks it against the evidence, but at runtime it is only a
  signal.
- The data is synthetic. The relationships the model learns (tenure
  in particular) need to be re-validated on real data.
