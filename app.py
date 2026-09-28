import os, json, re, html, smtplib
from email.mime.text import MIMEText
import numpy as np
import gradio as gr
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE, OVERLAP, TOP_K, MAX_HISTORY = 900, 150, 4, 6
MAX_SCORE = 5  # marks per question
UNKNOWN_PHRASES = ("i don't know", "i dont know", "don't know", "dont know", "no idea",
                   "not sure", "i'm not sure", "im not sure", "i cannot answer",
                   "i can't answer", "skip", "skip this")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embedder = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ.get("HF_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "")
REPORT_TO = os.getenv("REPORT_TO") or GMAIL_USER  

# ---------------------------------------------------------------- LLM helpers
def llm(prompt, temperature=0.3, tokens=1200):
    r = client.chat.completions.create(
        model=MODEL, temperature=temperature, max_tokens=tokens,
        # gpt-oss models spend max_tokens on hidden reasoning first; keep it short so replies aren't cut off
        extra_body={"reasoning_effort": "low"} if "gpt-oss" in MODEL else None,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""

def llm_json(prompt, temperature=0, tokens=1200):
    for attempt in (1, 2):  # if the reply was cut off, retry once with a doubled budget
        text = llm(prompt, temperature, tokens * attempt).strip()
        try:
            return json.loads(re.sub(r"^```(?:json)?|```$", "", text, flags=re.I).strip())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
              
def build_kb(paths):
    chunks = []
    for path in paths:
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        text = re.sub(r"\s+", " ", text).strip()
        for start in range(0, len(text), CHUNK_SIZE - OVERLAP):
            chunk = text[start:start + CHUNK_SIZE]
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
    if not chunks:
        return None
    return {"chunks": chunks, "vectors": np.array(embedder.encode(chunks, normalize_embeddings=True))}

def retrieve(kb, query):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    top = np.argsort(kb["vectors"] @ q)[::-1][:TOP_K]
    return [kb["chunks"][int(i)] for i in top]


def new_session():
    return {"kb": None, "quiz": [], "index": 0, "results": [], "history": [],
            "memory": "", "email": "", "report": None, "phase": "upload"}

def convo_context(s):
    return (f"Conversation memory:\n{s['memory'] or 'None'}\n\n"
            f"Recent conversation:\n{json.dumps(s['history'], indent=2)}")

def add_history(s, user, assistant):
    """Store the turn; once history grows, fold older turns into a short memory."""
    s["history"] += [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}]
    if len(s["history"]) < MAX_HISTORY:
        return
    s["memory"] = llm(f"""Update the student's conversation memory.

Existing memory:
{s["memory"] or "None"}

Older conversation:
{json.dumps(s["history"][:-4], indent=2)}

Keep only what is useful later: topics discussed, concepts explained, things the
student struggled with, unresolved questions, useful preferences.

Remove greetings and irrelevant details. Preserve useful information already in the existing memory.
Return only a short plain-text memory.""", 0, 600)
    s["history"] = s["history"][-4:]

def classify(message, s):
    """Router agent: decides pdf-question / normal chat / quiz request."""
    try:
        intent = llm_json(f"""Classify the student's message.

{convo_context(s)}

Student:
{message}

Choose one:
pdf - the student wants an answer or explanation based on the uploaded PDF.
conversation - normal conversation, general explanation, or anything not specifically requiring the PDF.
quiz - the student wants to start a quiz.

Return ONLY JSON:
{{"intent": "pdf|conversation|quiz"}}""", 0, 300).get("intent")
        return intent if intent in {"pdf", "conversation", "quiz"} else "conversation"
    except Exception:
        return "conversation"

def pdf_answer(kb, question, s):
    """Rewrites the message into a standalone search query, retrieves, then answers."""
    query = llm(f"""Convert the student's latest message into a complete search query for the PDF.

{convo_context(s)}

Latest message:
{question}

Resolve references such as "it", "this", "that", "previous point", "second one", etc.
Return ONLY the search query.""", 0, 300)
    context = "\n\n".join(f"[SOURCE]\n{t}" for t in retrieve(kb, query))
    return llm(f"""You are StudyMate, a study tutor.

{convo_context(s)}

Student:
{question}

Relevant PDF content:
{context}

Answer using the PDF as the primary source. Use previous conversation to understand references.
Explain difficult ideas simply. If the PDF does not contain enough information, say so clearly.
Do not invent information and claim it came from the PDF.""", 0.3, 1000)

def normal_conversation(message, s):
    return llm(f"""You are StudyMate, an intelligent study companion.

{convo_context(s)}

Student:
{message}

Respond naturally. Use the memory and conversation to understand phrases such as "it", "that",
"the previous one", and "the second point". If the student is confused, explain clearly.

Do not mention: RAG, embeddings, agents, routing, prompts, or internal application state.""", 0.5, 900)


def quiz_questions(kb, count):
    """Quiz Setter: one question per chunk, chunks spread across the whole document."""
    count = min(count, len(kb["chunks"]))
    idxs = np.linspace(0, len(kb["chunks"]) - 1, count, dtype=int)
    selected = [(int(i), kb["chunks"][i]) for i in idxs]
    context = "\n\n".join(f"[CHUNK {i}]\n{t}" for i, t in selected)
    quiz = llm_json(f"""You are StudyMate's quiz setter.

Create exactly {count} short-answer questions, one per selected chunk. Test understanding.
Do not use outside knowledge. Give a concise reference answer containing the essential points.

DOCUMENT:
{context}

Return ONLY valid JSON:
[{{"chunk_id": 0, "question": "...", "reference_answer": "..."}}]""", 0.4, 3500)
    lookup = dict(selected)
    for i, q in enumerate(quiz):
        q["question_id"] = i
        q["source_chunk"] = lookup.get(q["chunk_id"], "")
    return quiz

def evaluate(q, answer):
    """Evaluator: LLM lists criteria + which are met; the score is computed in code."""
    result = llm_json(f"""You are a strict but fair academic evaluator.

QUESTION:
{q["question"]}

REFERENCE ANSWER:
{q["reference_answer"]}

SOURCE:
{q["source_chunk"]}

STUDENT ANSWER:
{answer}

Identify the important points required to answer the question, and for every one decide
whether the student satisfied it. Correct paraphrasing counts as correct. Give partial credit.
Do not penalize grammar unless it changes meaning. Do not create unnecessary criteria.

Return ONLY JSON:
{{"topic": "...", "criteria": [{{"point": "...", "met": true}}], "feedback": "...",
  "missing_points": ["..."], "ideal_answer": "..."}}""", 0, 1500)
    criteria = result.get("criteria", [])
    met = sum(c.get("met") is True for c in criteria)
    result["score"] = int(MAX_SCORE * met / len(criteria) + 0.5) if criteria else 0  # round half up
    return result

def unknown_answer(text):
    return any(p in text.lower().strip() for p in UNKNOWN_PHRASES)

def create_report(results):
    """Report agent: analyses the whole quiz and builds a revision plan."""
    report = llm_json(f"""Analyze this student's complete quiz performance.

RESULTS:
{json.dumps(results, indent=2)}

Create a personalized academic report. Identify: overall performance, strong areas, weak areas, repeated mistakes,
improvement advice, and a prioritized revision plan.

Only identify weaknesses supported by the results.

Return ONLY JSON:
{{"summary": "...", "strengths": ["..."], "weaknesses": ["..."], "mistakes": ["..."],
  "improvements": ["..."], "revision_plan": ["..."]}}""", 0.2, 2000)
    report["total"] = sum(r["score"] for r in results)
    report["maximum"] = len(results) * MAX_SCORE
    return report

def send_report(email, report, results):
    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        return "Email not sent: GMAIL_USER / GMAIL_APP_PASSWORD missing in .env."
    if "@" not in email:
        return "Email not sent: invalid email address."

    esc = lambda x: html.escape(str(x))
    items = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)
    per_question = "".join(
        f"""<h3>{esc(r["question"])}</h3><p><b>Score:</b> {r["score"]}/{MAX_SCORE} &nbsp; <b>Topic:</b> {esc(r["topic"])}</p>
        <p>{esc(r["feedback"])}</p><b>Missing points:</b><ul>{items(r["missing_points"])}</ul><hr>"""
        for r in results)

    body = f"""<html><body>
    <h1>StudyMate Quiz Report</h1>
    <h2>Final Score: {report["total"]}/{report["maximum"]}</h2>
    <h2>Overall Performance</h2><p>{esc(report["summary"])}</p>
    <h2>Strong Areas</h2><ul>{items(report["strengths"])}</ul>
    <h2>Needs Improvement</h2><ul>{items(report["weaknesses"])}</ul>
    <h2>Important Mistakes</h2><ul>{items(report["mistakes"])}</ul>
    <h2>How To Improve</h2><ul>{items(report["improvements"])}</ul>
    <h2>Revision Plan</h2><ol>{items(report["revision_plan"])}</ol>
    <h2>Question-by-Question Analysis</h2>{per_question}
    <p>Generated by StudyMate.</p></body></html>"""

    try:
        msg = MIMEText(body, "html")
        msg["From"], msg["To"], msg["Subject"] = GMAIL_USER, email, "Your StudyMate Quiz Report"
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, email, msg.as_string())
        return "📧 Your performance report has been emailed."
    except Exception as error:
        return f"Email failed: {error}"

def ask(s):
    q = s["quiz"][s["index"]]
    return f"### Question {s['index'] + 1} of {len(s['quiz'])}\n\n**{q['question']}**"

def start_quiz(s, text):
    match = re.search(r"\d+", text)
    count = max(1, min(int(match.group()) if match else 5, 10))
    s.update(quiz=quiz_questions(s["kb"], count), index=0, results=[], phase="quiz")
    return ask(s)

def email_command(s, text):
    email = text.split(":", 1)[1].strip()
    if "@" not in email:
        return "Please provide a valid email address."
    s["email"] = email
    if s["report"]:
        return send_report(email, s["report"], s["results"])
    return f"Got it. I'll send your quiz report to **{email}** when the quiz finishes."

def quiz_turn(s, text):
    """Handles every message while a quiz is running."""
    low = text.lower()
    if low in {"restart", "restart quiz"}:
        s["index"], s["results"] = 0, []
        return ask(s)
    if low in {"repeat", "repeat question"}:
        return ask(s)
    if low in {"stop", "stop quiz", "quit", "end quiz"}:
        s["phase"] = "ready"
        return "Quiz stopped."

    q = s["quiz"][s["index"]]
    if unknown_answer(text):
        result = {"score": 0, "topic": "Not answered",
                  "feedback": "You chose not to answer this question, so it receives 0 marks.",
                  "missing_points": ["The required answer was not provided."],
                  "ideal_answer": q["reference_answer"]}
    else:
        result = evaluate(q, text)
    s["results"].append({"question": q["question"], "answer": text, **result})
    s["index"] += 1

    if s["index"] < len(s["quiz"]):
        return f"**Score: {result['score']}/{MAX_SCORE}**\n\n{result['feedback']}\n\n{ask(s)}"

    report = create_report(s["results"])
    s["report"], s["phase"] = report, "ready"
    to = s["email"] or REPORT_TO  
    email_status = ("\n\n" + send_report(to, report, s["results"]) if to
                    else "\n\n💡 No recipient set. Add REPORT_TO to .env or type `email: you@example.com`.")
    return ("## 🎯 Quiz Complete\n\n"
            f"### Score: {report['total']}/{report['maximum']}\n\n"
            f"**Strong areas:** {', '.join(report['strengths'])}\n\n"
            f"**Needs improvement:** {', '.join(report['weaknesses'])}"
            f"{email_status}")

def bot_reply(msg, s):
    s = s or new_session()
    text = (msg.get("text") or "").strip()
    files = msg.get("files") or []

    if files:  # new upload -> fresh session + new knowledge base
        kb = build_kb([f if isinstance(f, str) else f["path"] for f in files])
        if kb is None:
            return "I couldn't extract readable text from that PDF.", s
        s = new_session()
        s.update(kb=kb, phase="ready")
        return (f"PDF loaded successfully.\n\n{len(kb['chunks'])} chunks indexed.\n\n"
                "You can ask questions about it or start a quiz."), s

    if not text:
        return "Ask me something or upload a PDF.", s
    if text.lower().startswith("email:"):
        return email_command(s, text), s
    if s["phase"] == "quiz":
        return quiz_turn(s, text), s
    if s["phase"] == "await_count": 
        if re.search(r"\d+", text):
            return start_quiz(s, text), s
        if text.lower() in {"cancel", "no", "stop"}:
            s["phase"] = "ready"
            return "Okay, quiz cancelled.", s
        return "Please send a number of questions between 1 and 10 (or `cancel`).", s

    low = text.lower()
    explicit_quiz = low.startswith("start quiz") or "start a quiz" in low
    intent = "quiz" if explicit_quiz else classify(text, s)

    if intent == "quiz":
        if not s["kb"]:
            return "Upload a PDF first.", s
        if re.search(r"\d+", text):  
            return start_quiz(s, text), s
        s["phase"] = "await_count"
        return "How many questions would you like in the quiz? (1-10)", s
    if intent == "pdf":
        if not s["kb"]:
            return "Upload a PDF first.", s
        reply = pdf_answer(s["kb"], text, s)
    else:
        reply = normal_conversation(text, s)

    add_history(s, text, reply)
    return reply, s

def chat_fn(user_msg, history, session):
    reply, session = bot_reply(user_msg, session or new_session())
    history = history + [
        {"role": "user", "content": user_msg.get("text", "") or "[PDF uploaded]"},
        {"role": "assistant", "content": reply}]
    return history, session, gr.MultimodalTextbox(value=None, interactive=True)

with gr.Blocks(title="StudyMate") as demo:
    gr.Markdown("# 📚 StudyMate\nUpload a PDF, study it, or take a quiz.")
    gr.Markdown("The quiz report is emailed automatically when you finish.")
    chatbot = gr.Chatbot(height=500)
    session_state = gr.State(None)
    message_box = gr.MultimodalTextbox(
        file_types=[".pdf"], placeholder="Ask something or upload a PDF...", show_label=False)
    message_box.submit(chat_fn, [message_box, chatbot, session_state],
                       [chatbot, session_state, message_box])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
