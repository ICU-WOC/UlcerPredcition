# Ulcer Prediction (Vercel + Next.js)

ICU 입실 환자의 욕창 발생 위험도를 5가지 nurse-centered 변수로 추정하는 웹 도구.
GPT-4o 기반 RAG 챗봇 위젯이 페이지 우하단에 함께 임베드되어 사용자의 도구 관련 질문에
답변합니다.

- **프런트엔드**: Next.js 14 (Pages Router) + TypeScript + Bootstrap
- **백엔드**: Vercel Python serverless functions
  - `api/predict.py` — 욕창 위험도 예측
  - `api/chat.py` — RAG 챗봇 (GPT-4o + text-embedding-3-small + Supabase 로깅)
- **모델**: Gaussian Naïve Bayes + Isotonic recalibration
- **RAG**: 사전 임베딩(`data/embeddings.json`) + cold-start 인메모리 코사인 검색

## 디렉터리 구조

```
.
├── api/
│   ├── predict.py             # 욕창 예측 serverless 함수
│   ├── chat.py                # RAG 챗봇 serverless 함수
│   ├── calibrated_model.py    # joblib unpickle 시 필요한 래퍼 클래스
│   └── ulcer_prediction.pkl   # 학습된 모델 (3KB, imblearn 의존성 제거됨)
├── components/
│   └── ChatWidget.tsx         # 우하단 플로팅 챗봇 위젯
├── data/
│   └── embeddings.json        # RAG 사전 임베딩 (~3.9MB, Vercel 함수에 includeFiles 됨)
├── pages/
│   ├── _app.tsx
│   └── index.tsx              # 메인 페이지 (예측 폼 + FAQ + ChatWidget)
├── public/
│   └── chatbot.png            # 플로팅 버튼 아이콘
├── rag/                       # RAG 원본 문서 (gitignored — 미출판 원고 보호)
│   ├── Manuscript.docx
│   ├── analysis.ipynb
│   ├── calibrated_model.py
│   ├── predict.py
│   └── faq.md
├── scripts/
│   └── build_index.py         # RAG 임베딩 재빌드 스크립트
├── styles/globals.css
├── package.json
├── tsconfig.json
├── next.config.js
├── requirements.txt           # Python deps for Vercel (sklearn, openai 등)
└── vercel.json                # 함수별 메모리/timeout/includeFiles 설정
```

## 환경변수

Vercel Project Settings 와 로컬 `.env.local` 양쪽에 등록 필요:

| 변수 | 용도 |
|---|---|
| `OPENAI_API_KEY` | RAG 임베딩 + 챗 응답 생성 (text-embedding-3-small / gpt-4o) |
| `SUPABASE_URL` | 대화 로그 저장용 Supabase 프로젝트 URL |
| `SUPABASE_SERVICE_ROLE_KEY` | RLS 우회 INSERT (서버 전용 — 절대 클라이언트로 노출 금지) |

`.env.local` 은 `.gitignore` 로 보호되며 커밋되지 않습니다. Vercel 환경변수는
대시보드 → Settings → Environment Variables 에서 Production / Preview / Development
모두 등록 후 redeploy 해야 적용됩니다.

## 변수 기여도 계산 방식 (`/api/predict`)

기존 Flask 앱은 `shap.KernelExplainer` 와 background CSV 를 사용했지만,
Vercel Hobby 의 50MB 함수 크기 제한 때문에 `shap` + `pandas` + `numba`
런타임 의존성(약 80MB) 을 빼고 **Naïve Bayes 의 분석적 marginal contribution**
으로 대체했습니다. NB 는 additive 모델이라 marginal 이 정확한 Shapley 값과
일치하므로 출력 의미는 동일합니다 (KernelExplainer 의 sampling 근사 vs 정확값).

## 입력 변수 순서

1. `feature1` 의식수준 (RASS, −5 ~ +4)
2. `feature2` 가장 낮은 체온 (℃)
3. `feature3` 가장 높은 체온 (℃)
4. `feature4` 하지근력 (0 ~ 10)
5. `feature5` 하루 평균 실금 횟수

## 출력 점수 해석

학습 데이터의 자연 prevalence (3.58%) 와 isotonic recalibration 의 결과로
모델의 raw 확률 출력 상한이 약 23.5% 입니다.
사용자 친화적 표시를 위해 다음과 같이 0~100 점수로 매핑합니다:

- raw 0%   → 0점
- raw 10% → 50점 (임상 임계값)
- raw 23.5% → 100점

따라서 화면의 점수 50 이상이면 적극적 예방 중재 검토 대상으로 해석.

## RAG 챗봇 (`/api/chat`)

`pages/index.tsx` 우하단의 플로팅 버튼을 클릭하면 챗 패널이 열리고,
사용자의 질문이 `/api/chat` 으로 POST 됩니다.

**파이프라인**:
1. 클라이언트가 `localStorage` 의 anon UUID 와 함께 메시지 송신
2. `api/chat.py` 가 직전 사용자 + 어시스턴트 턴까지 합쳐 retrieval query 구성
3. `text-embedding-3-small` 로 임베딩 → `data/embeddings.json` 의 사전 임베딩과 코사인 유사도 비교
4. Top-6 청크 + 시스템 프롬프트 + 대화 히스토리를 GPT-4o 에 전달
5. 답변 송신 직후 Supabase `chat_logs` 테이블에 best-effort INSERT (실패해도 응답엔 영향 없음)

**Supabase 스키마**:
```sql
create table chat_logs (
  id          bigint generated always as identity primary key,
  user_id     text        not null,         -- 브라우저별 anon UUID
  created_at  timestamptz not null default now(),
  user_msg    text        not null,
  bot_reply   text        not null
);
create index on chat_logs (user_id, created_at);
```

**스코프 제한**:
시스템 프롬프트가 욕창 도구 관련 질문 외엔 거절하도록 강하게 제어하며,
RAG 코사인 점수가 `SIMILARITY_THRESHOLD` (0.18) 미만이면
"해당 정보는 자료에 없습니다. 부서 WOC에게 문의" 메시지로 응답합니다.

## RAG 임베딩 재빌드

`rag/` 의 원본 문서를 수정한 경우:

```bash
pip install -r scripts/requirements.txt
python scripts/build_index.py
```

→ 새 `data/embeddings.json` 이 생성됨 (약 3.9MB).
이 파일은 git 에 커밋되어 Vercel 함수에 함께 배포됩니다.

소스별 청크 크기는 `scripts/build_index.py` 의 `MAX_CHARS_BY_NAME` 에 정의:
- `Manuscript.docx`: 1000자
- `faq.md`: 600자 (Q/A 한 쌍이 한 청크가 되도록)
- 그 외: 1500자

## 로컬 개발

```bash
npm install
npm run dev
# http://localhost:3000
```

`next dev` 는 Python 함수를 실행하지 않으므로 위젯 UI 만 확인 가능.
실제 `/api/chat`, `/api/predict` 호출까지 로컬 검증하려면:

```bash
npm i -g vercel
vercel dev
```

## Vercel 배포

1. GitHub repo 에 push (main 브랜치에 머지 시 자동 배포)
2. https://vercel.com/new 에서 repo 선택
3. Framework Preset: **Next.js** (자동 감지)
4. Root Directory: `./`
5. Environment Variables 등록 (위 환경변수 표 참고)
6. Deploy

`requirements.txt` 가 자동 인식되어 Python 함수에 sklearn/scipy/numpy/joblib/openai
가 설치됩니다. Cold start 는 모델/임베딩 로드로 1~3초 소요될 수 있음.

## 모델 재학습 / 재생성

학습 노트북·전처리 스크립트는 별도 개발 저장소에서 관리됩니다.
새 pkl 만 교체하려면 `api/ulcer_prediction.pkl` 만 덮어쓰고 다시 push 하면 됩니다.
(joblib 으로 저장된 `CalibratedUlcerModel` 인스턴스 — `api/calibrated_model.py` 의
클래스 정의를 변경하지 않는 한 동일 형태로 유지)
