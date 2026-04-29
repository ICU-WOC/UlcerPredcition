import { useEffect, useRef, useState } from 'react';

type Role = 'user' | 'assistant';
type Message = { role: Role; content: string };

const GREETING: Message = {
  role: 'assistant',
  content:
    '안녕하세요. 욕창(Pressure Injury) 예측 모델 챗봇입니다.\n모델, 입력 변수, 성능, 사용 방법 등 자료에 있는 내용을 무엇이든 물어보세요.',
};

const USER_ID_KEY = 'ulcer_chat_user_id';
const DEFAULT_API = '/api/chat';

function getOrCreateUserId(): string {
  if (typeof window === 'undefined') return 'anonymous';
  let id = window.localStorage.getItem(USER_ID_KEY);
  if (!id) {
    id =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `u_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
    window.localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}

export default function ChatWidget() {
  const apiUrl = process.env.NEXT_PUBLIC_CHAT_API_URL || DEFAULT_API;

  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const userIdRef = useRef<string>('anonymous');
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    userIdRef.current = getOrCreateUserId();
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  async function send() {
    const content = input.trim();
    if (!content || loading) return;
    const next: Message[] = [...messages, { role: 'user', content }];
    setMessages(next);
    setInput('');
    setLoading(true);
    try {
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: next, user_id: userIdRef.current }),
      });
      const data = await res.json();
      const reply: string = data.error
        ? `오류가 발생했습니다: ${data.error}`
        : data.reply || '(빈 응답)';
      setMessages([...next, { role: 'assistant', content: reply }]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages([...next, { role: 'assistant', content: `네트워크 오류: ${msg}` }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <>
      {!open && (
        <div className="chat-fab-wrap">
          <span className="chat-fab-label" aria-hidden="true">
            여기를 눌러 상담하세요
          </span>
          <button
            aria-label="상담 챗봇 열기"
            onClick={() => setOpen(true)}
            className="chat-fab"
          >
            <span className="chat-fab-ring" aria-hidden="true" />
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="34"
              height="34"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </button>
        </div>
      )}

      {open && (
        <div className="chat-panel" role="dialog" aria-label="욕창 예측 챗봇">
          <div className="chat-header">
            <div>
              <div className="chat-title">욕창 예측 모델 챗봇</div>
              <div className="chat-subtitle">아산병원 RAG · GPT-4o</div>
            </div>
            <button
              aria-label="닫기"
              onClick={() => setOpen(false)}
              className="chat-close"
            >
              ×
            </button>
          </div>

          <div ref={scrollRef} className="chat-body">
            {messages.map((m, i) => (
              <Bubble key={i} role={m.role} content={m.content} />
            ))}
            {loading && <Bubble role="assistant" content="생각 중…" muted />}
          </div>

          <div className="chat-input-row">
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="질문을 입력하고 Enter (줄바꿈은 Shift+Enter)"
              disabled={loading}
              className="chat-input"
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              className="chat-send"
            >
              전송
            </button>
          </div>
        </div>
      )}
    </>
  );
}

function Bubble({
  role,
  content,
  muted,
}: {
  role: Role;
  content: string;
  muted?: boolean;
}) {
  const isUser = role === 'user';
  return (
    <div className={`chat-row ${isUser ? 'is-user' : 'is-bot'}`}>
      <div className={`chat-bubble ${isUser ? 'b-user' : 'b-bot'} ${muted ? 'is-muted' : ''}`}>
        {content}
      </div>
    </div>
  );
}
