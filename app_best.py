import os
import json
import re
import html
import logging
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from pypdf import PdfReader
from groq import Groq
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
OVERLAP = int(os.getenv("OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_HISTORY = 6
MAX_SCORE = 5

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("studymate")

UNKNOWN_PHRASES = (
    "i don't know", "i dont know", "don't know", "dont know",
    "no idea", "not sure", "i'm not sure", "im not sure",
    "i cannot answer", "i can't answer", "skip", "skip this"
)

# -----------------------------
# Safe clients
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
REPORT_TO = os.getenv("REPORT_TO") or os.getenv("GMAIL_USER")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None


def configuration_status():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    return missing


def llm(prompt, temperature=0.3, tokens=1200):
    if client is None:
        raise RuntimeError(
            "StudyMate is not configured: GROQ_API_KEY is missing."
        )
    kwargs = {
        "model": MODEL,
        "temperature": temperature,
        "max_tokens": tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if "gpt-oss" in MODEL:
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    result = client.chat.completions.create(**kwargs)
    return result.choices[0].message.content or ""


def llm_json(prompt, temperature=0, tokens=1200):
    last = ""
    for attempt in range(2):
        last = llm(prompt, temperature, tokens * (attempt + 1)).strip()
        cleaned = re.sub(
            r"^\s*```(?:json)?\s*|\s*```\s*$", "", last, flags=re.I
        ).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Recover the first JSON object/array if the model added prose.
            start_obj = min(
                [x for x in (cleaned.find("{"), cleaned.find("[")) if x >= 0],
                default=-1,
            )
            if start_obj >= 0:
                try:
                    return json.loads(cleaned[start_obj:])
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"Model returned invalid JSON: {last[:300]}")


# -----------------------------
# Embeddings / retrieval
# -----------------------------
def embed(texts):
    if not hf_client:
        raise RuntimeError(
            "StudyMate is not configured: HF_TOKEN is missing."
        )
    if isinstance(texts, str):
        texts = [texts]
    vectors = np.asarray(
        hf_client.feature_extraction(texts, model=EMBED_MODEL),
        dtype=np.float32,
    )

    # Some providers return nested arrays for a single request.
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def normalize_text(text):
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(text, page_number):
    text = normalize_text(text)
    if not text:
        return []

    step = max(1, CHUNK_SIZE - OVERLAP)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if len(chunk) >= 80:
            chunks.append({
                "text": chunk,
                "page": page_number,
            })
    return chunks


def build_kb(paths):
    chunks = []

    for path in paths:
        try:
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                chunks.extend(chunk_page(text, page_number))
        except Exception as exc:
            log.exception("PDF processing failed for %s", path)
            raise RuntimeError(
                f"Could not read '{Path(path).name}': {exc}"
            ) from exc

    if not chunks:
        return None

    vectors = embed([c["text"] for c in chunks])

    # Lightweight lexical index for hybrid retrieval.
    for chunk in chunks:
        words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b",
                               chunk["text"].lower()))
        chunk["terms"] = words

    return {
        "chunks": chunks,
        "vectors": vectors,
        "files": [Path(p).name for p in paths],
    }


def lexical_score(query, chunk):
    q_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))
    if not q_terms:
        return 0.0
    overlap = len(q_terms & chunk["terms"])
    return overlap / len(q_terms)


def retrieve(kb, query, top_k=TOP_K):
    q_vector = embed([query])[0]
    semantic = kb["vectors"] @ q_vector

    # Hybrid score: semantic retrieval + a small lexical signal.
    lexical = np.array(
        [lexical_score(query, c) for c in kb["chunks"]],
        dtype=np.float32,
    )
    score = 0.82 * semantic + 0.18 * lexical

    # Diverse page-aware selection.
    ranked = np.argsort(score)[::-1]
    chosen = []
    pages_seen = set()

    for idx in ranked:
        idx = int(idx)
        page = kb["chunks"][idx]["page"]
        if page not in pages_seen or len(chosen) < max(2, top_k // 2):
            chosen.append({
                **kb["chunks"][idx],
                "score": float(score[idx]),
                "chunk_index": idx,
            })
            pages_seen.add(page)
        if len(chosen) >= top_k:
            break

    return chosen


# -----------------------------
# Session + memory
# -----------------------------
def new_session():
    return {
        "kb": None,
        "quiz": [],
        "index": 0,
        "results": [],
        "history": [],
        "memory": "",
        "email": "",
        "report": None,
        "phase": "upload",
        "stats": {"questions": 0, "score": 0},
    }


def convo_context(s):
    return (
        f"Conversation memory:\n{s.get('memory') or 'None'}\n\n"
        f"Recent conversation:\n"
        f"{json.dumps(s.get('history', []), indent=2)}"
    )


def add_history(s, user, assistant):
    s["history"].extend([
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ])

    if len(s["history"]) < MAX_HISTORY:
        return

    try:
        s["memory"] = llm(
            f"""Update the student's short study memory.

Existing memory:
{s["memory"] or "None"}

Older conversation:
{json.dumps(s["history"][:-4], indent=2)}

Keep only durable study context:
- topics discussed
- concepts the student struggled with
- unresolved questions
- useful learning preferences

Remove greetings and irrelevant details.
Return only short plain text.""",
            0,
            500,
        )
    except Exception:
        log.exception("Memory update failed")

    s["history"] = s["history"][-4:]


# -----------------------------
# Study tutor
# -----------------------------
def classify(message, s):
    try:
        result = llm_json(
            f"""Classify the student's message.

{convo_context(s)}

Student:
{message}

Choose exactly one:
pdf = requires the uploaded PDF
conversation = general study conversation
quiz = asks to start/take a quiz

Return only:
{{"intent":"pdf|conversation|quiz"}}""",
            0,
            250,
        )
        intent = result.get("intent")
        return intent if intent in {"pdf", "conversation", "quiz"} else "conversation"
    except Exception:
        return "conversation"


def pdf_answer(kb, question, s):
    query = llm(
        f"""Rewrite the student's latest message into a precise standalone
search query for the uploaded study document.

{convo_context(s)}

Student:
{question}

Resolve references such as "it", "this", "that", "second one", etc.
Return only the search query.""",
        0,
        250,
    )

    sources = retrieve(kb, query)
    context = "\n\n".join(
        f"[PDF SOURCE | Page {x['page']}]\n{x['text']}"
        for x in sources
    )

    answer = llm(
        f"""You are StudyMate, a high-quality academic tutor.

{convo_context(s)}

Student:
{question}

Relevant PDF content:
{context}

Rules:
1. Use the PDF as the primary source.
2. Do not invent facts and attribute them to the PDF.
3. If the PDF is insufficient, say that clearly.
4. Explain difficult concepts in simple, exam-friendly language.
5. Give a concise answer first, then explanation when useful.
6. At the end add a "Sources" line listing the PDF page numbers used.
7. Do not mention RAG, embeddings, agents, prompts, routing, or internal code.""",
        0.25,
        1200,
    )

    page_list = sorted({x["page"] for x in sources})
    if page_list and "Sources" not in answer:
        answer += f"\n\n**Sources:** PDF pages {', '.join(map(str, page_list))}"
    return answer


def normal_conversation(message, s):
    return llm(
        f"""You are StudyMate, an intelligent academic study companion.

{convo_context(s)}

Student:
{message}

Respond naturally and helpfully. Explain concepts clearly.
When appropriate, structure answers with headings, bullets, examples,
and exam-ready points.

Do not mention internal application details such as RAG, embeddings,
agents, routing, or prompts.""",
        0.45,
        1000,
    )


# -----------------------------
# Quiz engine
# -----------------------------
def quiz_questions(kb, count, difficulty="Medium"):
    count = max(1, min(int(count), 15))

    # Spread questions across the document.
    indices = np.linspace(
        0, len(kb["chunks"]) - 1, count, dtype=int
    )
    selected = [(int(i), kb["chunks"][int(i)]) for i in indices]
    context = "\n\n".join(
        f"[CHUNK {i} | PAGE {c['page']}]\n{c['text']}"
        for i, c in selected
    )

    quiz = llm_json(
        f"""You are StudyMate's academic quiz setter.

Create exactly {count} short-answer questions from the supplied document.
Difficulty: {difficulty}.
Test understanding rather than copying sentences.
Do not use outside knowledge.

For every question provide:
- question
- reference_answer
- topic
- source_page

Return ONLY valid JSON:
[
  {{
    "question":"...",
    "reference_answer":"...",
    "topic":"...",
    "source_page":1
  }}
]

DOCUMENT:
{context}""",
        0.35,
        4000,
    )

    return quiz


def evaluate(q, answer):
    result = llm_json(
        f"""You are a strict but fair academic evaluator.

QUESTION:
{q["question"]}

REFERENCE ANSWER:
{q["reference_answer"]}

STUDENT ANSWER:
{answer}

Evaluate based only on the reference answer.
Correct paraphrasing counts.
Grammar should not reduce marks unless it changes meaning.

Return ONLY:
{{
  "criteria":[{{"point":"...", "met":true}}],
  "feedback":"...",
  "missing_points":["..."],
  "ideal_answer":"..."
}}""",
        0,
        1600,
    )

    criteria = result.get("criteria", [])
    met = sum(c.get("met") is True for c in criteria)
    result["score"] = int(
        MAX_SCORE * met / len(criteria) + 0.5
    ) if criteria else 0
    result["topic"] = q.get("topic", "General")
    return result


def unknown_answer(text):
    return text.lower().strip() in UNKNOWN_PHRASES


def create_report(results):
    report = llm_json(
        f"""Analyze this student's quiz performance.

RESULTS:
{json.dumps(results, indent=2)}

Create a useful study report. Identify:
- overall performance
- strong areas
- weak areas
- repeated mistakes
- improvement advice
- prioritized revision plan

Only identify weaknesses supported by the results.

Return ONLY:
{{
 "summary":"...",
 "strengths":["..."],
 "weaknesses":["..."],
 "mistakes":["..."],
 "improvements":["..."],
 "revision_plan":["..."]
}}""",
        0.2,
        2200,
    )

    report["total"] = sum(r["score"] for r in results)
    report["maximum"] = len(results) * MAX_SCORE
    return report


# -----------------------------
# Email report
# -----------------------------
def send_report(email, report, results):
    if not RESEND_API_KEY:
        return "📧 Report generated. Email is disabled because RESEND_API_KEY is missing."
    if "@" not in email:
        return "Email not sent: invalid email address."

    esc = lambda x: html.escape(str(x))
    items = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)

    per_question = "".join(
        f"""<h3>{esc(r["question"])}</h3>
        <p><b>Score:</b> {r["score"]}/{MAX_SCORE}
        &nbsp; <b>Topic:</b> {esc(r["topic"])}</p>
        <p>{esc(r["feedback"])}</p>
        <b>Missing points:</b><ul>{items(r.get("missing_points", []))}</ul>
        <hr>"""
        for r in results
    )

    body = f"""<html><body>
    <h1>📚 StudyMate Quiz Report</h1>
    <h2>Final Score: {report["total"]}/{report["maximum"]}</h2>
    <h2>Overall Performance</h2><p>{esc(report["summary"])}</p>
    <h2>Strong Areas</h2><ul>{items(report["strengths"])}</ul>
    <h2>Needs Improvement</h2><ul>{items(report["weaknesses"])}</ul>
    <h2>Important Mistakes</h2><ul>{items(report["mistakes"])}</ul>
    <h2>How To Improve</h2><ul>{items(report["improvements"])}</ul>
    <h2>Revision Plan</h2><ol>{items(report["revision_plan"])}</ol>
    <h2>Question-by-Question Analysis</h2>{per_question}
    <p>Generated by StudyMate.</p>
    </body></html>"""

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": "StudyMate <onboarding@resend.dev>",
                "to": [email],
                "subject": "Your StudyMate Quiz Report",
                "html": body,
            },
            timeout=20,
        )
        if response.status_code >= 300:
            return f"Email failed: {response.text[:300]}"
        return "📧 Your performance report has been emailed."
    except Exception as exc:
        log.exception("Email failed")
        return f"Email failed: {exc}"


# -----------------------------
# Chat / quiz flow
# -----------------------------
def ask(s):
    q = s["quiz"][s["index"]]
    return (
        f"### Question {s['index'] + 1} of {len(s['quiz'])}\n\n"
        f"**{q['question']}**\n\n"
        f"*Topic: {q.get('topic', 'General')}*"
    )


def start_quiz(s, count, difficulty="Medium"):
    s.update(
        quiz=quiz_questions(s["kb"], count, difficulty),
        index=0,
        results=[],
        phase="quiz",
    )
    return ask(s)


def quiz_turn(s, text):
    low = text.lower().strip()

    if low in {"restart", "restart quiz"}:
        s["index"], s["results"] = 0, []
        return ask(s)

    if low in {"repeat", "repeat question"}:
        return ask(s)

    if low in {"stop", "stop quiz", "quit", "end quiz"}:
        s["phase"] = "ready"
        return "Quiz stopped. You can ask another question or start a new quiz."

    q = s["quiz"][s["index"]]

    if unknown_answer(text):
        result = {
            "score": 0,
            "topic": q.get("topic", "Not answered"),
            "feedback": "No answer was provided, so this question receives 0 marks.",
            "missing_points": ["The required answer was not provided."],
            "ideal_answer": q["reference_answer"],
        }
    else:
        result = evaluate(q, text)

    s["results"].append({
        "question": q["question"],
        "answer": text,
        **result,
    })
    s["index"] += 1

    if s["index"] < len(s["quiz"]):
        return (
            f"**Score: {result['score']}/{MAX_SCORE}**\n\n"
            f"{result['feedback']}\n\n---\n\n{ask(s)}"
        )

    report = create_report(s["results"])
    s["report"], s["phase"] = report, "ready"

    recipient = s["email"] or REPORT_TO
    email_status = ""
    if recipient:
        email_status = "\n\n" + send_report(
            recipient, report, s["results"]
        )

    percentage = (
        round(report["total"] / report["maximum"] * 100)
        if report["maximum"] else 0
    )

    return (
        "## 🎯 Quiz Complete\n\n"
        f"### Score: {report['total']}/{report['maximum']} ({percentage}%)\n\n"
        f"**Strong areas:** {', '.join(report['strengths']) or 'None identified'}\n\n"
        f"**Needs improvement:** {', '.join(report['weaknesses']) or 'None identified'}"
        f"{email_status}"
    )


def upload_pdf(files, s):
    if not files:
        return "Please upload a PDF first.", s

    try:
        paths = [
            f if isinstance(f, str) else f["path"]
            for f in files
        ]
        kb = build_kb(paths)
        if kb is None:
            return "I couldn't extract readable text from the PDF.", s

        s = new_session()
        s.update(kb=kb, phase="ready")

        pages = len({
            c["page"] for c in kb["chunks"]
        })

        return (
            "## ✅ PDF Ready\n\n"
            f"**Files:** {', '.join(kb['files'])}\n\n"
            f"**Pages indexed:** {pages}\n\n"
            f"**Knowledge chunks:** {len(kb['chunks'])}\n\n"
            "You can now **ask questions**, **start a quiz**, or use **Exam Prep**."
        ), s
    except Exception as exc:
        return f"❌ PDF processing failed: {exc}", s


def bot_reply(msg, s):
    s = s or new_session()
    text = (msg.get("text") or "").strip()
    files = msg.get("files") or []

    if files:
        return upload_pdf(files, s)

    if not text:
        return "Ask me something or upload a PDF.", s

    if text.lower().startswith("email:"):
        email = text.split(":", 1)[1].strip()
        if "@" not in email:
            return "Please provide a valid email address.", s
        s["email"] = email
        if s["report"]:
            return send_report(email, s["report"], s["results"]), s
        return f"Got it. I'll send the report to **{email}** when the quiz finishes.", s

    if s["phase"] == "quiz":
        reply = quiz_turn(s, text)
        return reply, s

    low = text.lower()

    if low.startswith("start quiz") or "start a quiz" in low:
        if not s["kb"]:
            return "Upload a PDF first.", s

        match = re.search(r"\d+", text)
        count = max(1, min(int(match.group()) if match else 5, 15))

        difficulty = "Hard" if "hard" in low else (
            "Easy" if "easy" in low else "Medium"
        )
        return start_quiz(s, count, difficulty), s

    if low in {"help", "menu", "commands"}:
        return (
            "### 🧭 StudyMate Commands\n\n"
            "- **Ask a question** → answers from your PDF\n"
            "- **Start quiz** → creates a quiz\n"
            "- **Start quiz 10 hard** → 10-question hard quiz\n"
            "- **Repeat question** → repeats the current quiz question\n"
            "- **Restart quiz** → restarts the quiz\n"
            "- **Stop quiz** → ends the quiz\n"
            "- **email: you@example.com** → set report recipient\n\n"
            "You can also just talk naturally."
        ), s

    if not s["kb"]:
        reply = normal_conversation(text, s)
    else:
        intent = classify(text, s)
        reply = (
            pdf_answer(s["kb"], text, s)
            if intent == "pdf"
            else normal_conversation(text, s)
        )

    add_history(s, text, reply)
    return reply, s


def chat_fn(user_msg, history, session):
    reply, session = bot_reply(
        user_msg,
        session or new_session(),
    )

    history = history or []
    history = history + [
        {
            "role": "user",
            "content": user_msg.get("text", "") or "📄 PDF uploaded",
        },
        {
            "role": "assistant",
            "content": reply,
        },
    ]

    return history, session, gr.MultimodalTextbox(
        value=None,
        interactive=True,
    )


# -----------------------------
# Render / health
# -----------------------------
def health_check():
    missing = configuration_status()
    if missing:
        return f"StudyMate configuration incomplete: missing {', '.join(missing)}"
    return "StudyMate is healthy."


# -----------------------------
# Premium StudyMate GUI
# -----------------------------
CUSTOM_CSS = """
/* Overall app */
.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 18px 28px 35px !important;
}

/* Hero */
#hero {
    border: 1px solid rgba(120,120,140,.18);
    border-radius: 28px;
    padding: 30px 34px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(14,165,233,.07));
}
#hero h1 {
    font-size: 42px !important;
    margin-bottom: 4px !important;
}
.hero-sub {
    font-size: 16px;
    opacity: .72;
}

/* Cards */
.card {
    border: 1px solid rgba(120,120,140,.18);
    border-radius: 20px;
    padding: 18px !important;
}
.stat-card {
    text-align: center;
    min-height: 105px;
}
.stat-number {
    font-size: 28px;
    font-weight: 750;
}
.stat-label {
    opacity: .65;
    font-size: 13px;
}

/* Chat */
#chat {
    border-radius: 22px !important;
    border: 1px solid rgba(120,120,140,.18) !important;
}
#input {
    border-radius: 18px !important;
}

/* Sidebar */
#sidebar {
    min-width: 285px;
}
.sidebar-title {
    font-weight: 700;
    font-size: 17px;
}
.small {
    opacity: .68;
    font-size: 13px;
}

/* Buttons */
button {
    border-radius: 12px !important;
}

/* Footer */
.footer {
    text-align: center;
    opacity: .55;
    font-size: 12px;
    padding-top: 18px;
}

/* Mobile */
@media (max-width: 800px) {
    .gradio-container {
        padding: 10px !important;
    }
    #hero h1 {
        font-size: 31px !important;
    }
}
"""

with gr.Blocks(
    title="StudyMate | AI Study Companion",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=CUSTOM_CSS,
) as demo:

    # ---------- Hero ----------
    gr.Markdown(
        """
<div id="hero">
<h1>📚 StudyMate</h1>
<div class="hero-sub">
Your AI-powered study companion — <b>learn smarter, practice better, score higher.</b>
</div>
</div>
""",
        elem_id="hero",
    )

    # ---------- Top stats ----------
    with gr.Row():
        with gr.Column(elem_classes=["card", "stat-card"]):
            gr.Markdown(
                '<div class="stat-number">📄</div><div class="stat-label">PDF Tutor</div>'
            )
        with gr.Column(elem_classes=["card", "stat-card"]):
            gr.Markdown(
                '<div class="stat-number">🧠</div><div class="stat-label">AI Explanations</div>'
            )
        with gr.Column(elem_classes=["card", "stat-card"]):
            gr.Markdown(
                '<div class="stat-number">📝</div><div class="stat-label">Smart Quizzes</div>'
            )
        with gr.Column(elem_classes=["card", "stat-card"]):
            gr.Markdown(
                '<div class="stat-number">📊</div><div class="stat-label">Performance Analysis</div>'
            )

    with gr.Row(equal_height=False):

        # ---------- Left sidebar ----------
        with gr.Column(scale=1, elem_id="sidebar"):

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 📖 Study Material")
                gr.Markdown(
                    "Upload one or more PDFs. StudyMate will index them and use them as your study source.",
                    elem_classes=["small"],
                )

                upload_box = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )

                load_button = gr.Button(
                    "🚀 Load Study Material",
                    variant="primary",
                    size="lg",
                )

                file_status = gr.Markdown(
                    "No document loaded.",
                    elem_classes=["small"],
                )

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### ⚡ Quick Actions")

                quiz_5 = gr.Button("📝 5 Question Quiz")
                quiz_10 = gr.Button("🎯 10 Question Quiz")
                hard_quiz = gr.Button("🔥 Hard Quiz")
                help_button = gr.Button("🧭 StudyMate Help")

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 💡 Try asking")

                examples = [
                    "Explain the main concept simply.",
                    "Give me an exam-ready answer.",
                    "What are the important topics?",
                    "Explain the second point with an example.",
                    "Start quiz 10 hard",
                ]

                for example in examples:
                    gr.Button(
                        example,
                        size="sm",
                    )

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 📧 Quiz Report")
                email_box = gr.Textbox(
                    label="Email",
                    placeholder="you@example.com",
                    info="Used for your quiz performance report.",
                )
                save_email = gr.Button("💾 Save Email", size="sm")
                email_status = gr.Markdown(elem_classes=["small"])

        # ---------- Main content ----------
        with gr.Column(scale=3):

            status = gr.Markdown(
                "🟢 **Ready to study** — upload a PDF to begin.",
                elem_classes=["card"],
            )

            chatbot = gr.Chatbot(
                height=610,
                type="messages",
                elem_id="chat",
                placeholder="""
<div style="text-align:center;padding:60px 20px">
<h2>👋 Welcome to StudyMate</h2>
<p>Upload your study PDF and start asking questions.</p>
<br>
<p>Try: <b>“Explain this chapter in simple words.”</b></p>
</div>
""",
            )

            message_box = gr.MultimodalTextbox(
                file_types=[".pdf"],
                file_count="multiple",
                placeholder="💬 Ask anything about your PDF…",
                show_label=False,
                elem_id="input",
            )

            with gr.Row():
                clear_button = gr.ClearButton(
                    [message_box, chatbot],
                    value="🗑️ Clear Chat",
                )
                repeat_button = gr.Button(
                    "🔁 Repeat Question",
                    size="sm",
                )
                restart_button = gr.Button(
                    "🔄 Restart Quiz",
                    size="sm",
                )

            gr.Markdown(
                """
<div class="footer">
StudyMate • RAG-powered AI learning assistant • Built for smarter exam preparation
</div>
"""
            )

    session_state = gr.State(new_session())

    # ---------- Core chat ----------
    message_box.submit(
        chat_fn,
        [message_box, chatbot, session_state],
        [chatbot, session_state, message_box],
    )

    # ---------- Upload ----------
    def load_from_sidebar(files, session):
        if not files:
            return "⚠️ Please select a PDF first.", session, "No document loaded."

        reply, session = upload_pdf(files, session or new_session())
        return reply, session, "🟢 **Study material loaded successfully**"

    load_button.click(
        load_from_sidebar,
        [upload_box, session_state],
        [file_status, session_state, status],
    )

    # ---------- Quick quiz actions ----------
    def quick_quiz(n, difficulty, session):
        session = session or new_session()
        if not session["kb"]:
            return (
                [{"role": "assistant", "content": "⚠️ Upload a PDF first."}],
                session,
            )
        reply = start_quiz(session, n, difficulty)
        return [{"role": "assistant", "content": reply}], session

    quiz_5.click(
        lambda s: quick_quiz(5, "Medium", s),
        [session_state],
        [chatbot, session_state],
    )

    quiz_10.click(
        lambda s: quick_quiz(10, "Medium", s),
        [session_state],
        [chatbot, session_state],
    )

    hard_quiz.click(
        lambda s: quick_quiz(10, "Hard", s),
        [session_state],
        [chatbot, session_state],
    )

    help_button.click(
        lambda: [{"role": "assistant", "content":
            """### 🧭 StudyMate Help

**Ask your PDF**
- “What is software engineering?”
- “Explain this in simple words.”
- “Give me a 10-mark answer.”

**Quiz**
- `start quiz`
- `start quiz 10`
- `start quiz 10 hard`

**During a quiz**
- `repeat question`
- `restart quiz`
- `stop quiz`

**Report**
- `email: you@example.com`

You can also simply chat naturally with StudyMate."""
        }],
        outputs=chatbot,
    )

    save_email.click(
        lambda email, s: (
            (s.update(email=email.strip()) or "✅ Email saved for the quiz report.")
            if email and "@" in email
            else "⚠️ Enter a valid email address."
        ),
        [email_box, session_state],
        [email_status],
    )

    repeat_button.click(
        lambda s: (
            [{"role": "assistant", "content": ask(s)}]
            if s and s.get("phase") == "quiz"
            else [{"role": "assistant", "content": "There is no active quiz."}]
        ),
        [session_state],
        [chatbot],
    )

    restart_button.click(
        lambda s: (
            (
                [{"role": "assistant", "content": ask(s)}],
                s
            )
            if s and s.get("quiz")
            else (
                [{"role": "assistant", "content": "There is no quiz to restart."}],
                s or new_session()
            )
        ),
        [session_state],
        [chatbot, session_state],
    )

    def initial_status():
        missing = configuration_status()
        if missing:
            return (
                "🟠 **Configuration needed:** "
                + ", ".join(missing)
            )
        return "🟢 **System online** — upload your study material to begin."

    demo.load(initial_status, outputs=status)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
