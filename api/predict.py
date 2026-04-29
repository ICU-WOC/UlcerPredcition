"""Vercel Python serverless 함수: /api/predict

5변수(의식수준, 최저체온, 최고체온, 하지근력, 실금횟수)를 받아
- raw 확률 (model 출력)
- 표시용 위험도 점수 (0~10% raw -> 0~50% display, 10~23.5% -> 50~100%)
- 변수별 위험 기여도 (NB 분석적 marginal contribution; shap 라이브러리 미사용)
를 반환한다.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calibrated_model import CalibratedUlcerModel  # noqa: E402, F401  joblib unpickle 용

import joblib  # noqa: E402

MODEL_PATH = os.path.join(HERE, 'ulcer_prediction.pkl')

# 학습 cap (isotonic 출력 상한). 임계값 10% (raw) 가 표시 50% 가 되도록 설계.
RAW_THRESHOLD = 0.10
RAW_CAP = 0.2355

# 화면에 표시할 변수 라벨 — 모델 입력 순서와 동일
LABELS = ['의식수준', '저체온', '고체온', '하지근력', '실금']

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def display_score(p_raw):
    """raw 확률(0~1)을 화면 표시용 0~100 점수로 매핑.

    - raw 0           -> 0
    - raw RAW_THRESHOLD (10%)  -> 50  (임상 임계값)
    - raw RAW_CAP (23.5%)      -> 100
    """
    p = max(0.0, min(float(p_raw), RAW_CAP))
    if p <= RAW_THRESHOLD:
        return 50.0 * p / RAW_THRESHOLD
    return 50.0 + 50.0 * (p - RAW_THRESHOLD) / (RAW_CAP - RAW_THRESHOLD)


def feature_contributions(model, x):
    """변수별 위험 기여도 (raw 확률 기준 marginal contribution).

    각 변수 i 에 대해:
        contrib_i = P(y=1|x) - P(y=1|x_i := mean)
    NB 는 additive 모델이라 marginal 이 정확한 Shapley 값과 일치한다.
    """
    p_full = float(model.predict_proba(x)[0, 1])
    scaler = model.pipeline.named_steps['scaler']
    contribs = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        x_neutral = x.copy()
        x_neutral[0, i] = scaler.mean_[i]  # 학습 평균으로 치환 -> 효과 제거
        p_without = float(model.predict_proba(x_neutral)[0, 1])
        contribs[i] = p_full - p_without
    return contribs, p_full


def build_response(features):
    """예측 + 기여도 계산 후 JSON 직렬화 가능 dict 반환."""
    f1, f2, f3, f4, f5 = features
    if not (-5 <= f1 <= 4):
        return {'error': '의식수준은 -5에서 +4 사이여야 합니다.'}
    if not (0 <= f4 <= 10):
        return {'error': '하지근력은 0에서 10 사이여야 합니다.'}

    model = get_model()
    X = np.array([[f1, f2, f3, f4, f5]], dtype=float)
    contribs, p_raw = feature_contributions(model, X)

    score = round(display_score(p_raw), 1)
    # 일단 raw 기여도 (확률 point) 로 모은 뒤, 양수만 추리고 상대 백분위로 환산.
    items = []
    for label, val, c in zip(LABELS, features, contribs):
        items.append({
            'feature': label,
            'value': float(val),
            '_raw': float(c),
        })

    # 저체온 / 고체온 중 기여도가 큰 쪽만 남김 (기존 로직 유지)
    low = next(it for it in items if it['feature'] == '저체온')
    high = next(it for it in items if it['feature'] == '고체온')
    drop = '고체온' if low['_raw'] >= high['_raw'] else '저체온'
    items = [it for it in items if it['feature'] != drop]

    # 음수(보호 요인) 제거 — 사용자에게는 위험을 끌어올린 변수만 보여줌
    positives = [it for it in items if it['_raw'] > 0]
    total = sum(it['_raw'] for it in positives)

    out_features = []
    if total > 0:
        for it in sorted(positives, key=lambda x: x['_raw'], reverse=True):
            out_features.append({
                'feature': it['feature'],
                'value': round(it['value'], 1),
                'contribution': round(it['_raw'] / total * 100, 1),
            })

    return {
        'risk_score': score,
        'raw_probability': round(p_raw * 100, 2),
        'features': out_features,
    }


class handler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get('content-length', 0))
            raw = self.rfile.read(length).decode('utf-8')
            body = json.loads(raw) if raw else {}
            features = [
                float(body['feature1']),
                float(body['feature2']),
                float(body['feature3']),
                float(body['feature4']),
                float(body['feature5']),
            ]
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self._json({'error': f'입력 파싱 실패: {e}'}, 400)

        try:
            payload = build_response(features)
            status = 400 if 'error' in payload else 200
            return self._json(payload, status)
        except Exception as e:
            return self._json({'error': f'서버 오류: {e}'}, 500)

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._set_cors()
        self.end_headers()
        self.wfile.write(body)
