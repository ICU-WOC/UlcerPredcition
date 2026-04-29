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
    "죄송합니다. 저는 욕창(Pressure Injury) 예측 모델 자료에 기반한 질문에만 "
    "답변할 수 있습니다. 모델, 입력 변수, 성능, 사용 방법 등에 대해 물어봐 주세요."
)

SYSTEM_PROMPT = """당신은 욕창(Pressure Injury) 예측 모델에 대해 안내하는 한국어 챗봇입니다.

지식 출처: 사용자에게 제공된 '참고 컨텍스트' 블록(연구 논문 원고, 분석 노트북 코드, Flask 백엔드 코드, 웹 페이지 HTML).

규칙:
1. 컨텍스트에 근거해서만 답변하세요. 컨텍스트에 없는 사실은 추측하지 말고 "해당 정보는 자료에 없습니다."라고 말하세요.
2. 답변 주제는 다음으로 한정합니다: 욕창(Pressure Injury), 예측 모델의 알고리즘/성능/검증, 입력 변수(RASS, 하지 근력, 체온, 실금 등), 모델 사용법, 데이터/연구 방법론, 웹 애플리케이션 동작.
3. 위 주제와 관련 없는 질문(일상 대화, 다른 의학 주제, 일반 상식, 코딩 일반론 등)에는 정중히 거절하세요: "죄송합니다. 저는 욕창 예측 모델 관련 질문에만 답변할 수 있습니다."
4. 답변은 한국어로, 임상/연구 맥락에 맞춰 정확하고 간결하게 작성하세요. 수치는 컨텍스트에 명시된 값을 그대로 인용하세요.
5. 변수 이름은 자료에 표기된 한국어 명칭을 그대로 쓰세요(예: "RASS 평균값", "motor strength LE 평균", "체온(최대)", "체온(최소)", "하루평균실금"). 코드 변수명(예: rass_mean) 형식으로 임의 변환하지 마세요.
6. 사용자가 "그럼?", "왜?", "더 있어?" 같이 짧은 후속 질문을 하면 직전까지의 대화 맥락을 고려해 답하세요. 맥락상 명백히 욕창 모델 관련이면 거절하지 말고 답변하세요.
7. 의학적 진단/처방을 직접 내리지 말고, 모델 결과는 참고용임을 필요 시 안내하세요.
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
