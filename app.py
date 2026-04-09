"""
app.py – Streamlit UI for the US Census Text-to-SQL Chat Agent.

Run:
    cd backend && streamlit run app.py
"""

import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

from agent import answer_question
from db import fetch_schema_metadata, close_connection
from retrieval import bootstrap_retriever

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="US Census AI Assistant",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ===== Global ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #111111;
}

/* White background */
.stApp {
    background: #ffffff;
    color: #111111;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] { background: transparent; }

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #f5f6f8;
    border-right: 1px solid #e0e0e0;
}
section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* ===== Chat messages ===== */
.stChatMessage {
    border-radius: 10px;
    padding: 4px 0;
    animation: fadeIn 0.2s ease-in;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }

/* User bubble */
[data-testid="chatMessage-user"] {
    background: #eef4ff;
    border: 1px solid #c8dcff;
    border-radius: 10px;
}

/* Assistant bubble */
[data-testid="chatMessage-assistant"] {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
}

/* Ensure all chat text is black */
.stChatMessage p, .stChatMessage span, .stChatMessage div {
    color: #111111 !important;
}

/* ===== SQL expander ===== */
.streamlit-expanderHeader {
    background: #f0f0f0 !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 8px !important;
    font-size: 0.82rem;
    color: #444444 !important;
}
.streamlit-expanderContent {
    background: #fafafa !important;
    border: 1px solid #d0d0d0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ===== Input box ===== */
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 1px solid #cccccc !important;
    border-radius: 10px !important;
    color: #111111 !important;
    font-family: 'Inter', sans-serif;
}

/* ===== Metric cards ===== */
[data-testid="stMetric"] {
    background: #f0f2f5;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetric"] * { color: #111111 !important; }

/* ===== Buttons ===== */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #cccccc;
    background: #ffffff;
    color: #111111;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #eef4ff;
    border-color: #4a90e2;
    color: #1a5fb4;
}

h1, h2, h3, h4, h5, h6 { color: #111111; }
p, span, label, div { color: #111111; }
a { color: #1a5fb4; }
code { color: #c0392b; background: #fdf2f2; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False

if "sql_history" not in st.session_state:
    st.session_state.sql_history = []   # parallel list to messages (None or SQL string)

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

if "response_times" not in st.session_state:
    st.session_state.response_times = []  # seconds per completed query


# ---------------------------------------------------------------------------
# Retriever bootstrap (runs once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _build_retriever():
    """Fetch schema once from Snowflake and build the hybrid index."""
    schema = fetch_schema_metadata()
    ret = bootstrap_retriever(schema)
    return ret


def ensure_retriever():
    if not st.session_state.retriever_ready:
        with st.spinner("🔍 Loading schema index …"):
            _build_retriever()
        st.session_state.retriever_ready = True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🇺🇸 US Census AI")
    st.markdown("*Ask anything about US population data.*")
    st.divider()

    # Metrics
    st.markdown("### 📊 Session Stats")
    avg_ms = (
        int(sum(st.session_state.response_times) / len(st.session_state.response_times) * 1000)
        if st.session_state.response_times else 0
    )
    col1, col2 = st.columns(2)
    col1.metric("Queries Run", st.session_state.turn_count)
    col2.metric("Avg Response", f"{avg_ms} ms" if avg_ms else "—")

    st.divider()

    # Example questions
    st.markdown("### 💡 Try asking …")
    examples = [
        "What is the total US population?",
        "Which state has the highest median household income?",
        "Show me the 5 most populous counties in Texas.",
        "What percentage of California is Hispanic or Latino?",
        "How does the education level compare between New York and Florida?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state._pending_example = ex

    st.divider()

    # New conversation
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.sql_history = []
        st.session_state.turn_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#555555'>Powered by OpenAI GPT-OSS 120B via OpenRouter<br>"
        "Data: US Open Census on Snowflake Marketplace</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🇺🇸 US Census Data Assistant")
st.markdown(
    "Ask natural-language questions about US population, demographics, income, "
    "housing, education, and more — powered by real Census data."
)
st.divider()

# Bootstrap retriever (first run only)
ensure_retriever()

# Replay conversation history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show SQL for assistant turns if available
        if msg["role"] == "assistant" and i < len(st.session_state.sql_history):
            sql = st.session_state.sql_history[i]
            if sql:
                with st.expander("🔎 View SQL", expanded=False):
                    st.code(sql, language="sql")

# Handle example button presses
if hasattr(st.session_state, "_pending_example"):
    user_input = st.session_state._pending_example
    del st.session_state._pending_example
else:
    user_input = st.chat_input("Ask a question about US Census data …", key="chat_input")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build history for LLM (exclude current message; it's added inside agent)
    llm_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude the just-appended one
    ]

    # Run agent pipeline with a status indicator
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("*⏳ Thinking …*")

        _t0 = time.perf_counter()
        answer, sql_used = answer_question(user_input, llm_history)
        _elapsed = time.perf_counter() - _t0
        st.session_state.response_times.append(_elapsed)

        status_placeholder.empty()
        st.markdown(answer)

        if sql_used:
            with st.expander("🔎 View SQL", expanded=False):
                st.code(sql_used, language="sql")

    # Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sql_history.extend(
        [None] * (len(st.session_state.messages) - 1 - len(st.session_state.sql_history))
    )
    st.session_state.sql_history.append(sql_used)
    st.session_state.turn_count += 1
    st.rerun()  # refresh sidebar metrics immediately
