"""Vercel Python Serverless Function: /api/chat

POST { "messages": [{ "role": "user"|"assistant", "content": str }, ...] }
-> { "reply": str } | { "error": str }
"""
from http.server import BaseHTTPRequestHandler
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.json"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 6
SIMILARITY_THRESHOLD = 0.18

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

REFUSAL = (
    "죄송합니다. 저는 가이드라인 자료에 기반한 질문에만 "
    "답변할 수 있습니다. 제가 답변하지 못하는 내용은 부서 WOC에게 문의해주세요."
)

SYSTEM_PROMPT = """You are a clinical AI assistant specialized in pressure injury (욕창) risk assessment for ICU nurses at a tertiary hospital (서울아산병원 성인 중환자실).

Goal: Provide accurate, concise, context-grounded answers strictly based on the provided '참고 컨텍스트' block (연구 논문 원고, 통계 분석 노트북, 모델·보정 코드, 사용자 FAQ). Audience is clinical nurses who need clear and actionable understanding.

Scope (only answer within these topics):
- 욕창(Pressure Injury) 일반
- 본 도구의 알고리즘·성능·내부 검증·SHAP 해석·확률 보정(isotonic regression)
- 입력 변수: 의식수준 RASS, 하지근력, 최고 체온, 최저 체온, 하루 평균 실금횟수 (의미·측정·해석)
- 도구 사용 방법, 연구 방법론, FAQ 항목

Requirements:
1. Ground every answer ONLY in the provided context. If information is missing, respond exactly: "해당 정보는 자료에 없습니다. 자세한 내용은 부서 WOC에게 문의해주세요."
2. For out-of-scope queries (일상 대화, 다른 의학 주제, 일반 상식, 코딩 일반론 등), respond exactly: "죄송합니다. 저는 가이드라인 자료에 기반한 질문에만 답변할 수 있습니다. 제가 답변하지 못하는 내용은 부서 WOC에게 문의해주세요."
3. Use Korean. Keep answers precise, clinically relevant, and concise (3–6 sentences or ≤5 bullets).
4. Preserve exact terminology and numeric values (AUC, Recall, %, 95% CI 등) as written in context. Do not round or infer.
5. Maintain original Korean variable names exactly (예: "RASS", "하지근력", "최고 체온", "최저 체온", "하루 평균 실금횟수"). Do not convert to code-style names (예: rass_mean).
6. For short follow-ups ("왜?", "그럼?", "더 있어?"), use prior conversation context when clearly related to the tool.
7. Include safety note when relevant: this tool supports clinical decisions and does not replace clinical judgment.
8. When discussing performance or generalizability, note: single-center (서울아산병원 성인 ICU) data, internally validated, external validation not yet performed.

Constraints:
- Format: short paragraphs or bullet points
- Style: clinical, clear, concise, non-speculative
- Scope: no external knowledge, no assumptions beyond context
- Reasoning: think step-by-step internally, output only the final answer
- Self-check: verify all claims are grounded in context before answering

Take a deep breath and work through each question step-by-step.
"""


def _load_docs() -> list[dict]:
    with EMBEDDINGS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# Load once per cold start.
DOCS = _load_docs()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _retrieve(query_embedding: list[float], k: int) -> list[tuple[float, dict]]:
    scored = [(_cosine(query_embedding, d["embedding"]), d) for d in DOCS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def _last_user_message(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


def _build_retrieval_query(messages: list[dict]) -> str:
    """Combine the last user turn with prior context so short follow-ups
    ("그럼?", "왜?", "더 있어?") still retrieve relevant chunks.
    """
    last_user = _last_user_message(messages) or ""
    prior_user = ""
    seen_last = False
    for m in reversed(messages):
        if m.get("role") != "user" or not m.get("content"):
            continue
        if not seen_last:
            seen_last = True
            continue
        prior_user = m["content"]
        break
    last_assistant = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant = m["content"]
            break
    parts = [p for p in (prior_user, last_assistant[:400], last_user) if p]
    return "\n".join(parts) if parts else last_user


def _generate_reply(messages: list[dict]) -> str:
    user_query = _last_user_message(messages)
    if not user_query:
        return "질문을 입력해 주세요."

    client = OpenAI()

    retrieval_query = _build_retrieval_query(messages)
    emb = client.embeddings.create(model=EMBED_MODEL, input=retrieval_query).data[0].embedding
    top = _retrieve(emb, TOP_K)
    best_score = top[0][0] if top else 0.0

    if best_score < SIMILARITY_THRESHOLD:
        return REFUSAL

    context = "\n\n---\n\n".join(
        f"[출처: {doc['source']}]\n{doc['text']}" for _, doc in top
    )

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"참고 컨텍스트:\n{context}"},
            *[{"role": m["role"], "content": m["content"]} for m in messages],
        ],
    )
    return chat.choices[0].message.content or ""


def _log_to_supabase(user_id: str, user_msg: str, bot_reply: str) -> None:
    if not (SUPABASE_URL and SUPABASE_KEY):
        return
    payload = json.dumps([{
        "user_id": user_id,
        "user_msg": user_msg,
        "bot_reply": bot_reply,
    }]).encode("utf-8")
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/chat_logs",
        data=payload,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[supabase log failed] {type(e).__name__}: {e}", file=sys.stderr)


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send(400, {"error": "messages must be a non-empty array"})
            user_id = str(body.get("user_id") or "anonymous")
            reply = _generate_reply(messages)
            self._send(200, {"reply": reply})
            user_msg = _last_user_message(messages) or ""
            _log_to_supabase(user_id, user_msg, reply)
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
