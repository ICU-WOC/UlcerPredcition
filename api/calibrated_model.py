"""SMOTE pipeline + Dal Pozzolo prior correction + isotonic recalibration 래퍼.

Vercel serverless 배포용 — 학습 시점의 SMOTE 단계는 빠진 inference 전용
sklearn Pipeline 을 받습니다 (imblearn 런타임 의존성 제거 목적).

feature_names_in_ 순서 (앱 폼 순서):
    [RASS 평균값, 체온(최소), 체온(최대), motor strength LE 평균, 실금횟수총합]
"""

import numpy as np


class CalibratedUlcerModel:
    def __init__(self, pipeline, prior_ratio, iso_reg, feature_names):
        self.pipeline = pipeline
        self.r = prior_ratio
        self.iso_reg = iso_reg
        self.feature_names_in_ = np.array(feature_names)

    def predict_proba(self, X):
        p_raw = self.pipeline.predict_proba(X)[:, 1]
        p_pc = (p_raw * self.r) / (p_raw * self.r + (1 - p_raw))
        p_iso = self.iso_reg.transform(p_pc)
        return np.column_stack([1 - p_iso, p_iso])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
