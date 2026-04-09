# Conversational Agent — US Census Data

A natural-language AI agent that answers questions about US population, demographics, income, housing, and education by generating and executing SQL against the [US Open Census Data](https://app.snowflake.com/marketplace/listing/GZSTZ491VXY) dataset on Snowflake Marketplace.

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐     off-topic     ┌──────────────────────┐
│  Guardrail  │ ────────────────► │  Rejection message   │
└─────────────┘                   └──────────────────────┘
      │ on-topic
      ▼
┌──────────────────────────────────────────┐
│  Hybrid Retriever (schema context)       │
│  ├── BM25 (sparse / exact keyword)       │
│  ├── Neural Dense (all-MiniLM-L6-v2)    │
│  └── Reciprocal Rank Fusion (RRF)        │
└──────────────────────────────────────────┘
      │ top-k schema entries
      ▼
┌──────────────────────────────────────────┐
│  ReAct Agent Loop (up to 5 steps)        │
│  ├── THOUGHT → SQL → run on Snowflake   │
│  ├── Inspect results, refine query       │
│  └── FINAL_ANSWER                        │
└──────────────────────────────────────────┘
      │
      ▼
  Streamlit UI
```

**Key components:**

| File | Responsibility |
|---|---|
| `agent.py` | Guardrail, ReAct reasoning loop, final synthesis |
| `retrieval.py` | Hybrid Neural Dense + BM25 schema retriever |
| `db.py` | Snowflake connection singleton, query execution, schema inspector |
| `app.py` | Streamlit chat UI |

---

## Prerequisites

- Python 3.9+
- A [Snowflake](https://www.snowflake.com/) account with the **US Open Census Data — Neighborhood Insights (Free Dataset)** listing installed from the Marketplace
- An [OpenRouter](https://openrouter.ai/) API key

---

## 1. Clone the repository

```bash
git clone https://github.com/nithin-pulla/conversational-agent-us-census.git
cd conversational-agent-us-census
```

---

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

On first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB). It is cached locally after that.

---

## 4. Set up environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set each value:

```env
# ── Snowflake ──────────────────────────────────────────────────────────────
# Account identifier — found in Snowflake under Admin → Accounts
# Format: <orgname>-<accountname>  (e.g. myorg-xy12345)
SNOWFLAKE_ACCOUNT=your_account_identifier

# Snowflake login credentials
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password

# The database name as it appears after installing the Marketplace listing.
# Default name assigned by Snowflake:
SNOWFLAKE_DATABASE=US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET

SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_ROLE=ACCOUNTADMIN

# ── OpenRouter ─────────────────────────────────────────────────────────────
# Get your key at https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key

# ── Optional ───────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
```

> **Note:** `.env` is listed in `.gitignore` and will never be committed.

---

## 5. Verify Snowflake connectivity (optional but recommended)

```bash
python - <<'EOF'
from db import run_query
rows = run_query('SELECT CURRENT_USER(), CURRENT_DATABASE()', max_rows=1)
print(rows)
EOF
```

Expected output: a single row with your Snowflake username and the census database name.

---

## 6. Run the tests

Unit tests (no Snowflake or network required):

```bash
pytest tests/ -m "not integration" -v
```

Integration tests (requires live Snowflake connection):

```bash
pytest tests/ -m integration -v
```

---

## 7. Start the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first load it fetches schema metadata from Snowflake and builds the hybrid retrieval index — this takes ~15–30 seconds. Subsequent questions are served from the in-memory index.

---

## Example questions

- What is the total US population?
- Which state has the highest median household income?
- Show me the 5 most populous counties in Texas.
- What percentage of California is Hispanic or Latino?
- How does education level compare between New York and Florida?
- Are people rich in Manhattan?

---

## Retrieval approach

The schema retriever uses **Hybrid Neural Dense + BM25** with Reciprocal Rank Fusion:

- **BM25 (sparse):** Exact keyword matching over tokenised schema descriptions. Fast and precise for column codes (`B19013e1`) and known census terms.
- **Neural Dense (all-MiniLM-L6-v2):** Sentence-transformer embeddings that map user language into a semantic space, bridging vocabulary gaps such as `"home ownership"` → `"owner occupied tenure"` or `"commute time"` → `"travel time to work"`.
- **RRF Fusion:** Both ranked lists are merged using Reciprocal Rank Fusion (`k=60`) — no score normalisation required.

If `sentence-transformers` cannot be loaded, the retriever degrades gracefully to BM25-only mode.

---

## Project structure

```
.
├── agent.py          # ReAct agent: guardrail, reasoning loop, synthesis
├── app.py            # Streamlit chat UI
├── db.py             # Snowflake connection and schema inspector
├── retrieval.py      # Hybrid Neural Dense + BM25 retriever
├── requirements.txt  # Python dependencies
├── pytest.ini        # Test configuration
├── .env.example      # Environment variable template
└── tests/
    ├── conftest.py
    ├── test_agent.py
    ├── test_db.py
    ├── test_retrieval.py
    └── test_integration.py
```

---

## Troubleshooting

**`KeyError: SNOWFLAKE_ACCOUNT`**
Your `.env` file is missing or not in the working directory. Make sure you ran `cp .env.example .env` and filled in all values before starting.

**`OperationalError: 250001` (Snowflake connection refused)**
Check your `SNOWFLAKE_ACCOUNT` identifier format. It should be `<orgname>-<accountname>`, not a URL.

**`OPENROUTER_API_KEY` rate limit / 429**
The agent retries up to 3 times with exponential backoff. If errors persist, check your OpenRouter usage limits or switch to a different model in `agent.py`.

**Slow first response**
The first question triggers schema fetch + index build + model inference. Subsequent questions are fast (~2–5 s) because the index is cached in memory by Streamlit's `@st.cache_resource`.

**`sentence-transformers` not found**
Run `pip install sentence-transformers`. The retriever will warn and fall back to BM25-only mode without crashing.
