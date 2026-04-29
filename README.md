# Ulcer Prediction (Vercel + Next.js)

ICU 입실 환자의 욕창 발생 위험도를 5가지 nurse-centered 변수로 추정하는 웹 도구.

- 프론트엔드: Next.js 14 (Pages Router) + TypeScript
- 백엔드: Vercel Python serverless function (`api/predict.py`)
- 모델: Gaussian Naïve Bayes + Dal Pozzolo prior correction + Isotonic recalibration

## 디렉터리 구조

```
.
├── api/
│   ├── predict.py             # Vercel serverless 함수
│   ├── calibrated_model.py    # joblib unpickle 시 필요한 래퍼 클래스
│   └── ulcer_prediction.pkl   # 학습된 모델 (3KB, imblearn 의존성 제거됨)
├── pages/
│   ├── _app.tsx
│   └── index.tsx              # 메인 페이지
├── styles/globals.css
├── package.json
├── tsconfig.json
├── next.config.js
├── requirements.txt           # Python deps for Vercel
└── vercel.json
```

## 변수 기여도 계산 방식

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
- raw 10% → 50점 (DCA 기반 임상 임계값)
- raw 23.5% → 100점

따라서 화면의 점수 50 이상이면 적극적 예방 중재 검토 대상으로 해석.

## 로컬 개발

```bash
npm install
npm run dev
# http://localhost:3000
```

Vercel CLI 로 Python 함수 포함 로컬 테스트:
```bash
npm i -g vercel
vercel dev
```

## Vercel 배포

1. GitHub repo 에 push
2. https://vercel.com/new 에서 repo 선택
3. Framework Preset: **Next.js** (자동 감지)
4. Root Directory: `./`
5. Deploy

`requirements.txt` 가 자동 인식되어 Python 함수에 sklearn/scipy/numpy/joblib
이 설치됩니다. 첫 cold start 는 모델 로드 때문에 1~3초 소요될 수 있음.

## 모델 재학습 / 재생성

학습 노트북·전처리 스크립트는 별도 개발 저장소에서 관리됩니다.
새 pkl 만 교체하려면 `api/ulcer_prediction.pkl` 만 덮어쓰고 다시 push 하면 됩니다.
(joblib 으로 저장된 `CalibratedUlcerModel` 인스턴스 — `api/calibrated_model.py` 의
클래스 정의를 변경하지 않는 한 동일 형태로 유지)
