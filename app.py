import os
import re
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="IP-SAKTI Sahayak",
    page_icon="⚖️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "data" / "knowledge.txt"

# Free-tier friendly models, tried in this order.
# If one is missing (404) or out of quota (429), the next one is used.
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
]


# -------------------------------------------------
# GEMINI CONNECTION
# -------------------------------------------------

def load_api_key():
    """Read the key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return str(key).strip()
    except Exception:
        pass
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(name)
        if key:
            return key.strip()
    return None


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return genai.Client(api_key=api_key)


API_KEY = load_api_key()
client = get_client(API_KEY) if API_KEY else None


# -------------------------------------------------
# KNOWLEDGE BASE + TF-IDF INDEX
# -------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_index():
    try:
        text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    except Exception:
        return [], None, None

    text = text.replace("\r\n", "\n")
    sections = re.split(r"(?m)^(?=SOURCE \d+\s*$)", text)

    items = []
    for section in sections:
        if not section.strip().startswith("SOURCE"):
            continue
        title = re.search(r"Title:\s*(.+)", section)
        url = re.search(r"URL:\s*(.+)", section)
        if title and url:
            items.append(
                {
                    "title": title.group(1).strip(),
                    "url": url.group(1).strip(),
                    "text": section.strip(),
                }
            )

    if not items:
        return [], None, None

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform([s["text"] for s in items])
    return items, vec, matrix


sources, vectorizer, document_matrix = build_index()


def retrieve_sources(question, top_k=3, min_score=0.05):
    if not sources or vectorizer is None:
        return []
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, document_matrix)[0]
    results = []
    for idx in sims.argsort()[::-1][:top_k]:
        if sims[idx] > min_score:
            results.append({**sources[idx], "score": float(sims[idx])})
    return results


# -------------------------------------------------
# GEMINI CALL WITH FALLBACKS + FRIENDLY ERRORS
# -------------------------------------------------

def error_code(exc):
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def generate_with_fallback(prompt):
    """
    Returns (text, model_used, error_message).
    Never raises - problems come back as a friendly error_message.
    """
    if client is None:
        return None, None, "no_key"

    # Try the model that worked last time first.
    order = list(MODEL_CANDIDATES)
    last_good = st.session_state.get("working_model")
    if last_good in order:
        order.remove(last_good)
        order.insert(0, last_good)

    last_problem = "unknown"

    for model in order:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                last_problem = "empty"
                continue
            st.session_state["working_model"] = model
            return text, model, None

        except Exception as exc:  # noqa: BLE001
            code = error_code(exc)
            if code in (400, 401, 403) and (
                "API key" in str(exc) or "API_KEY" in str(exc) or code in (401, 403)
            ):
                return None, None, "bad_key"
            if code == 404:
                last_problem = "not_found"      # model unavailable -> try next
            elif code == 429:
                last_problem = "quota"          # rate limit -> try next
            elif code in (500, 502, 503, 504):
                last_problem = "busy"           # Google busy -> try next
            else:
                last_problem = "other"
            continue

    return None, None, last_problem


FRIENDLY_ERRORS = {
    "no_key": (
        "The Gemini API key is not set. Add it to `.streamlit/secrets.toml` as "
        "`GEMINI_API_KEY = \"your-key\"` (get a free key at "
        "https://aistudio.google.com/apikey)."
    ),
    "bad_key": (
        "Gemini did not accept the API key. Please create a new free key at "
        "https://aistudio.google.com/apikey and update `.streamlit/secrets.toml`."
    ),
    "quota": (
        "The free Gemini quota is used up for the moment. "
        "Please wait a minute and try again."
    ),
    "busy": "Gemini is busy right now. Please try again in a few seconds.",
    "not_found": (
        "No configured Gemini model is available for this key. "
        "Update MODEL_CANDIDATES in app.py."
    ),
    "empty": "Gemini returned an empty answer. Please rephrase and try again.",
    "other": "Could not reach Gemini right now. Please try again shortly.",
    "unknown": "Could not reach Gemini right now. Please try again shortly.",
}


def translate_for_search(question):
    """Non-English questions are translated to English so TF-IDF search works."""
    if question.isascii() or client is None:
        return question
    text, _, err = generate_with_fallback(
        "Translate this question to English. Reply with only the translation:\n\n"
        + question
    )
    return text if text and not err else question


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("⚖️ IP-SAKTI Sahayak")
st.subheader(
    "Multilingual, Source-Grounded AI Assistant for Ayurveda IP & Regulatory Guidance"
)
st.write(
    "Ask questions about Intellectual Property, Ayurveda, "
    "Traditional Knowledge and regulatory concepts."
)
st.divider()

if client is None:
    st.warning(FRIENDLY_ERRORS["no_key"])
if not sources:
    st.warning("Knowledge base could not be loaded (check data/knowledge.txt).")


# -------------------------------------------------
# SAMPLE QUESTIONS (clickable)
# -------------------------------------------------

SAMPLES = [
    "Can traditional Ayurvedic knowledge be patented?",
    "What is defensive protection of traditional knowledge?",
    "What is the WIPO treaty on genetic resources?",
]


def set_sample(text):
    st.session_state["question"] = text


st.markdown("### 💡 Try a question")
cols = st.columns(len(SAMPLES))
for col, sample in zip(cols, SAMPLES):
    with col:
        st.button(sample, on_click=set_sample, args=(sample,), use_container_width=True)


# -------------------------------------------------
# INPUTS
# -------------------------------------------------

question = st.text_area(
    "🔎 Enter your question:",
    key="question",
    placeholder="Example: Can traditional Ayurvedic knowledge be patented?",
    height=120,
)

language = st.selectbox(
    "🌐 Choose your response language:",
    ["English", "Hindi", "Marathi", "Tamil", "Telugu"],
)


# -------------------------------------------------
# ASK
# -------------------------------------------------

if st.button("🚀 Ask IP-SAKTI Sahayak", type="primary"):

    if not question.strip():
        st.warning("Please enter a question first.")

    elif client is None:
        st.error(FRIENDLY_ERRORS["no_key"])

    elif not sources:
        st.error("Knowledge base could not be loaded.")

    else:
        with st.spinner("🔍 Searching trusted sources..."):
            retrieved = retrieve_sources(translate_for_search(question))

        if not retrieved:
            st.warning("⚠️ Insufficient source-grounded evidence")
            st.write(
                "I could not find enough relevant information in the current "
                "knowledge base to answer this question reliably."
            )
            st.info(
                "Please try a question related to Ayurveda, Traditional "
                "Knowledge, Intellectual Property, WIPO or regulatory guidance."
            )
        else:
            context = "\n\n".join(
                f"[Source {i + 1}]\n{item['text']}" for i, item in enumerate(retrieved)
            )

            prompt = f"""
You are IP-SAKTI Sahayak.

You are a multilingual, source-grounded AI assistant
for Intellectual Property and regulatory guidance related
to Ayurveda and Traditional Knowledge.

USER QUESTION:
{question}

RESPONSE LANGUAGE:
{language}

SOURCE MATERIAL:
{context}

IMPORTANT RULES:

1. Answer ONLY using the information contained in
   the SOURCE MATERIAL.

2. Do not invent laws, regulations, sections,
   policies, cases or legal conclusions.

3. If the source material is not enough to answer
   something, clearly say that the available sources
   do not provide enough information.

4. Explain the answer in simple language.

5. Do not claim to be a lawyer, patent agent,
   government authority or regulatory authority.

6. Mention when the user should verify information
   with an official authority or qualified professional.

7. This is informational guidance and not legal advice.

8. Do not make up citations.

Give a clear answer in the requested language.
"""

            with st.spinner("🤖 Generating source-grounded answer..."):
                answer, model_used, problem = generate_with_fallback(prompt)

            if problem:
                st.warning(FRIENDLY_ERRORS.get(problem, FRIENDLY_ERRORS["other"]))
                st.info(
                    "In the meantime, here are the most relevant passages from "
                    "the knowledge base:"
                )
            else:
                st.success("✅ Source-grounded answer generated")
                st.markdown("### 💡 Answer")
                st.write(answer)
                st.caption(f"Model: {model_used}")

                st.markdown("### 🛡️ Evidence Status")
                if len(retrieved) >= 2:
                    st.success(
                        "Strong source coverage — multiple relevant sources were retrieved."
                    )
                else:
                    st.warning(
                        "Limited source coverage — verify the information "
                        "with the official source."
                    )

            st.markdown("### 📚 Sources Used")
            for item in retrieved:
                st.markdown(f"**{item['title']}**")
                st.markdown(item["url"])

            with st.expander("🔍 View retrieved knowledge", expanded=bool(problem)):
                for i, item in enumerate(retrieved):
                    st.markdown(f"**Source {i + 1}: {item['title']}**")
                    st.write(item["text"])
                    st.divider()


# -------------------------------------------------
# DISCLAIMER / FOOTER
# -------------------------------------------------

st.divider()
st.info(
    "⚠️ Disclaimer: IP-SAKTI Sahayak provides informational, source-grounded "
    "guidance only. It does not constitute legal or regulatory advice. "
    "Always verify important matters using official government or "
    "international sources."
)
st.caption(
    "IP-SAKTI Sahayak | Multilingual AI assistant for Ayurveda IP and regulatory guidance"
)
