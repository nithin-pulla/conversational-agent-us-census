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

Before running or deploying, you need:

1. **Python 3.11** — matches the pinned runtime; earlier versions are not tested
2. **Snowflake account** with the **US Open Census Data — Neighborhood Insights (Free Dataset)** listing installed from the [Snowflake Marketplace](https://app.snowflake.com/marketplace/listing/GZSTZ491VXY)
3. **OpenRouter API key** — free tier available at [openrouter.ai/keys](https://openrouter.ai/keys)

---

## Local development

### 1. Clone the repository

```bash
git clone https://github.com/nithin-pulla/conversational-agent-us-census.git
cd conversational-agent-us-census
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv

source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On first run, `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB). It is cached locally after that.

> **Running tests?** Install dev dependencies instead:
> ```bash
> pip install -r requirements-dev.txt
> ```

### 4. Configure credentials

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set each variable:

```env
# ── Snowflake ──────────────────────────────────────────────────────────────
# Account identifier — found under Admin → Accounts in Snowflake
# Format: <orgname>-<accountname>  (e.g. myorg-xy12345)
SNOWFLAKE_ACCOUNT=your_account_identifier

SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password

# Database name assigned by Snowflake after installing the Marketplace listing
SNOWFLAKE_DATABASE=US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET

SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_ROLE=ACCOUNTADMIN

# ── OpenRouter ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY=your_openrouter_api_key

# ── Optional ───────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
```

> `.env` is in `.gitignore` and will never be committed.

### 5. Verify Snowflake connectivity (optional but recommended)

```bash
python - <<'EOF'
from db import run_query
rows = run_query('SELECT CURRENT_USER(), CURRENT_DATABASE()', max_rows=1)
print(rows)
EOF
```

Expected output: a single row with your Snowflake username and the census database name.

### 6. Run the tests

Unit tests (no Snowflake or network required):

```bash
pytest tests/ -m "not integration" -v
```

Integration tests (requires live Snowflake connection):

```bash
pytest tests/ -m integration -v
```

### 7. Start the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first load it fetches schema metadata from Snowflake and builds the hybrid retrieval index (~15–30 seconds). Subsequent questions are served from the cached in-memory index.

---

## Deployment

### Option A — Streamlit Community Cloud (recommended, free)

Streamlit Community Cloud deploys directly from a GitHub repository with no server setup.

1. **Push your clone to GitHub** (the repo must be accessible to your Streamlit account — public or private both work).

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **New app** and fill in:
   - **Repository:** your fork/clone
   - **Branch:** `master`
   - **Main file path:** `app.py`

4. Open **Advanced settings** and set **Python version** to `3.11` (matches `runtime.txt`).

5. Still in Advanced settings, open the **Secrets** panel and paste the following, substituting your real values:

   ```toml
   SNOWFLAKE_ACCOUNT   = "your_account_identifier"
   SNOWFLAKE_USER      = "your_username"
   SNOWFLAKE_PASSWORD  = "your_password"
   SNOWFLAKE_DATABASE  = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"
   SNOWFLAKE_SCHEMA    = "PUBLIC"
   SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
   SNOWFLAKE_ROLE      = "ACCOUNTADMIN"
   OPENROUTER_API_KEY  = "sk-or-v1-..."
   ```

   > Secrets are stored encrypted by Streamlit and are never exposed in logs or the UI.

6. Click **Deploy**. The first boot takes ~2–3 minutes to install dependencies and download the embedding model.

---

### Option B — Docker (Railway, Render, Fly.io, or any VPS)

A `Dockerfile` is included. These steps work on any platform that can run Docker.

#### Build and run locally

```bash
docker build -t census-app .

docker run -p 8501:8501 \
  -e SNOWFLAKE_ACCOUNT=your_account_identifier \
  -e SNOWFLAKE_USER=your_username \
  -e SNOWFLAKE_PASSWORD=your_password \
  -e SNOWFLAKE_DATABASE=US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET \
  -e SNOWFLAKE_SCHEMA=PUBLIC \
  -e SNOWFLAKE_WAREHOUSE=COMPUTE_WH \
  -e SNOWFLAKE_ROLE=ACCOUNTADMIN \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  census-app
```

Or pass all variables from your local `.env` file:

```bash
docker run -p 8501:8501 --env-file .env census-app
```

Open `http://localhost:8501`.

#### Deploy to Railway

1. Push the repo to GitHub.
2. Create a new project at [railway.app](https://railway.app) → **Deploy from GitHub repo**.
3. Railway auto-detects the `Dockerfile` and builds it.
4. Under **Variables**, add all the environment variables listed in the `docker run` example above.
5. Railway assigns a public URL once the deploy finishes.

#### Deploy to Render

1. Create a new **Web Service** at [render.com](https://render.com).
2. Connect your GitHub repo.
3. Set **Environment** to `Docker` — Render picks up the `Dockerfile` automatically.
4. Add environment variables under **Environment → Add Environment Variable**.
5. Set the port to `8501`.

#### Deploy to Fly.io

```bash
# Install the Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
fly launch          # follow prompts, set app name and region
fly secrets set \
  SNOWFLAKE_ACCOUNT=your_account_identifier \
  SNOWFLAKE_USER=your_username \
  SNOWFLAKE_PASSWORD=your_password \
  SNOWFLAKE_DATABASE=US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET \
  SNOWFLAKE_SCHEMA=PUBLIC \
  SNOWFLAKE_WAREHOUSE=COMPUTE_WH \
  SNOWFLAKE_ROLE=ACCOUNTADMIN \
  OPENROUTER_API_KEY=sk-or-v1-...
fly deploy
```

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
- **Neural Dense (all-MiniLM-L6-v2):** Sentence-transformer embeddings that bridge vocabulary gaps such as `"home ownership"` → `"owner occupied tenure"` or `"commute time"` → `"travel time to work"`.
- **RRF Fusion:** Both ranked lists are merged using Reciprocal Rank Fusion (`k=60`) — no score normalisation required.

If `sentence-transformers` cannot be loaded, the retriever degrades gracefully to BM25-only mode.

---

## Project structure

```
.
├── agent.py                      # ReAct agent: guardrail, reasoning loop, synthesis
├── app.py                        # Streamlit chat UI
├── db.py                         # Snowflake connection and schema inspector
├── retrieval.py                  # Hybrid Neural Dense + BM25 retriever
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt          # Adds pytest/pytest-mock for local testing
├── runtime.txt                   # Python 3.11 — read by Streamlit Cloud
├── Dockerfile                    # Container build for Docker-based deploys
├── .dockerignore                 # Excludes venv/secrets/tests from image
├── .env.example                  # Environment variable template
├── .streamlit/
│   ├── config.toml               # UI theme (light mode, colours)
│   └── secrets.toml.example      # Streamlit secrets template
└── tests/
    ├── conftest.py
    ├── test_agent.py
    ├── test_db.py
    ├── test_retrieval.py
    └── test_integration.py
```

---

## Troubleshooting

**`No matching distribution found for streamlit` (or any other package)**
Your pip is outdated and cannot see newer package versions. Upgrade it first, then retry:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**`KeyError: SNOWFLAKE_ACCOUNT`**
Your credentials are not set. Locally: check that `.env` exists and is filled in. On Streamlit Cloud: verify the secrets are pasted correctly in the Secrets panel.

**`OperationalError: 250001` (Snowflake connection refused)**
Check your `SNOWFLAKE_ACCOUNT` format. It must be `<orgname>-<accountname>` (e.g. `myorg-xy12345`), not a full URL.

**`OPENROUTER_API_KEY` rate limit / 429**
The agent retries up to 3 times with exponential backoff. If errors persist, check your OpenRouter usage dashboard or switch to a different model in `agent.py`.

**Slow first response**
The first question triggers schema fetch + index build + model inference. Subsequent questions are fast (~2–5 s) because the index is cached in memory by Streamlit's `@st.cache_resource`.

**`sentence-transformers` not found**
Run `pip install -r requirements.txt`. The retriever warns and falls back to BM25-only mode without crashing.

**Docker build fails on `torch`**
`torch` is pulled in as a transitive dependency of `sentence-transformers`. It is large (~2 GB) and the build may time out on low-resource machines. Increase Docker's memory limit or build on a machine with at least 4 GB of RAM available to Docker.
