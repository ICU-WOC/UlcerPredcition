"""Vercel Python serverless 함수: /api/predict

5변수(의식수준, 최저체온, 최고체온, 하지근력, 실금횟수)를 받아
- raw 확률 (model 출력)
- 표시용 위험도 점수 (0~10% raw -> 0~50% display, 10~23.5% -> 50~100%)
- 변수별 위험 기여도 (NB 분석적 marginal contribution을 표시 점수 단위로 변환;
  '+X점' 형태로 표기하기 위함. NB는 additive라 marginal = 정확한 Shapley)
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
    """변수별 위험 기여도를 표시 점수(0~100) 단위로 반환.

    각 변수 i 에 대해 학습 평균으로 치환했을 때 표시 점수가 얼마나 떨어지는지를
    계산한다 (display_score(p_full) - display_score(p_without)).
    NB는 additive라 raw 확률 기준 marginal이 정확한 Shapley 값과 일치하며,
    여기에 piecewise display_score를 한 번 더 적용해 사용자가 보는 점수와
    같은 단위로 표시한다.
    """
    p_full = float(model.predict_proba(x)[0, 1])
    score_full = display_score(p_full)
    scaler = model.pipeline.named_steps['scaler']
    contribs = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        x_neutral = x.copy()
        x_neutral[0, i] = scaler.mean_[i]  # 학습 평균으로 치환 -> 효과 제거
        p_without = float(model.predict_proba(x_neutral)[0, 1])
        contribs[i] = score_full - display_score(p_without)
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
    # 변수별 표시 점수 기여도 (이미 0~100 점수 단위)
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

    # 위험을 끌어올린 변수만 (양의 점수 기여도) 노출, 보호 요인은 표시 안 함
    positives = [it for it in items if it['_raw'] > 0]

    out_features = []
    for it in sorted(positives, key=lambda x: x['_raw'], reverse=True):
        out_features.append({
            'feature': it['feature'],
            'value': round(it['value'], 1),
            'contribution': round(it['_raw'], 1),
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
