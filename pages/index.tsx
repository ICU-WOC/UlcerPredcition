import Head from 'next/head';
import { FormEvent, useState } from 'react';
import ChatWidget from '../components/ChatWidget';

type Contribution = { feature: string; value: number; contribution: number };
type ApiOk = { risk_score: number; raw_probability: number; features: Contribution[] };
type ApiErr = { error: string };
type ApiResponse = ApiOk | ApiErr;

const FAQS: { q: string; a: string }[] = [
  {
    q: 'Q1. 욕창 발생 확률은 어떻게 계산되는 건가요?',
    a: `이 도구는 Gaussian Naïve Bayes 머신러닝 모델로 욕창 발생 확률을 계산합니다.

입력하신 5가지 값(의식수준, 하지근력, 최고 체온, 최저 체온, 하루 평균 실금횟수)을 모델에 넣으면 0~100% 사이의 욕창 발생 확률이 출력됩니다. 클래스 불균형 보정(SMOTE)으로 인한 확률 왜곡을 보완하기 위해 isotonic regression 재보정(recalibration) 을 적용한 확률을 표시합니다.

추후 학습 데이터가 갱신되면 더 적합한 알고리즘으로 교체될 수 있습니다.`,
  },
  {
    q: 'Q2. 각 변수의 기여도는 어떻게 계산되는 건가요?',
    a: `머신러닝 모델의 의사결정을 해석하는 SHAP (SHapley Additive exPlanations) 값을 사용했습니다. 예측 결과에 영향을 준 요소들을 SHAP 값으로 점수화한 뒤, 위험을 끌어올리는 방향(양의 SHAP)으로 작용한 변수들 사이의 상대 비율(%)로 표시합니다. 따라서 표시된 % 의 합은 항상 100% 가 됩니다.`,
  },
  {
    q: 'Q3. 이 프로그램은 얼마나 정확한가요?',
    a: `이 모델은 서울아산병원 성인 중환자실에 입실한 22,428명 환자 데이터로 학습 및 내부 검증되었으며, 정확도는 테스트 데이터 기준 87.7% (AUC 0.887, Recall 0.739) 로 측정되었습니다.

정확도를 이해하기 위해 임계점을 알아야 합니다. 임계점이란 쉽게 말해 머신러닝 모델이 해당 환자를 욕창군으로 판단하는 최소 위험도 값입니다.

예를 들어, A,B,C 환자의 욕창 위험도가 각각 30점, 40점, 50점이라고 해봅시다. 임계점을 30점으로 설정하면 30점 이상인 A, B, C 환자 모두 욕창 위험군으로 분류할 것입니다. 만약 임계점을 50점으로 설정한다면 C 환자만 욕창 위험군으로 분류됩니다.

Decision Curve Analysis(DCA) 결과 이 모델은 임계 확률 약 15% 이하 범위에서 임상적 이득(net benefit) 이 확인되었습니다. 즉 위험도가 다소 낮더라도 욕창 위험군으로 분류해 예방 중재를 시작하는 편이 환자에게 더 유익할 수 있습니다.

다만, 이 지표는 모델 학습에 사용한 데이터를 기준으로 측정되었기에 실제 임상현장에서 적용했을 때는 변동 가능합니다.`,
  },
  {
    q: 'Q4. 욕창에 영향을 미치는 다른 요인들도 있는 것으로 알고 있는데 왜 이 변수들만 선택한건가요?',
    a: `첫 프로토타입 모델 개발 당시 국내외 욕창을 키워드로 발간된 다수의 논문에서 p-value가 한 번이라도 유의한 결과를 보인 변수 중 AMIS 3.0에서 수집할 수 있는 것을 종합하여 헤모글로빈, 알부민, BMI 등 30여개의 변수를 분석하였습니다.

그러나 내부 논의를 거쳐 간호사가 직접 중재를 제공하며 욕창을 예방하는 활동에 참여할 수 있는 변수만 다시 선정하여 모델을 구축하였습니다. 최종 선정한 변수는 의식수준, 하지근력, 체온, 실금이며 모델의 정확도는 이 변수만으로 욕창을 예측했을 때 계산하였습니다.`,
  },
  {
    q: 'Q5. 알파고 딥러닝과 같은 매커니즘인가요?',
    a: `아니요. 딥러닝(인공신경망)은 머신러닝의 하위 분야이며 알파고가 사용한 모델 계열입니다. 이 도구는 Gaussian Naïve Bayes 모델을 사용합니다.

PyCaret 을 사용하여 Gradient Boosting, Random Forest, AdaBoost, LightGBM, SVM, Logistic Regression, Naïve Bayes 등 여러 알고리즘을 비교했는데, Gradient Boosting 이 AUC 는 가장 높았으나(0.894) Recall 이 0.09 에 그쳐 욕창 발생 환자를 거의 놓치는 한계가 있었습니다. 임상에서는 욕창 발생을 놓치지 않는 것이 더 중요하다고 판단하여 AUC·Recall·F1-score 의 균형이 가장 좋은 Gaussian Naïve Bayes 가 최종 선정되었습니다.

추후 데이터 갱신에 따라 더 적합한 알고리즘으로 교체될 수 있습니다.`,
  },
  {
    q: 'Q6. 저희가 값을 입력할 때마다 실시간으로 업데이트가 되는건가요?',
    a: `아니요. 개발한 모델은 실시간 업데이트가 지원되지 않습니다. 사전에 모델링에 사용할 수 있도록 추출한 환자 데이터를 사용하여 만든 머신러닝 모델에서 성능은 고정됩니다.

따라서 이후 모델이 예측하게 될 새로운 환자들의 특성이 변화함에 따라 정확도는 변화할 수 있습니다.`,
  },
];

function ResultCard({ data }: { data: ApiOk }) {
  return (
    <div className="mt-4 p-4 bg-white shadow rounded">
      <h4 className="text-black">
        <span style={{ display: 'block', textAlign: 'center', fontSize: 22, fontWeight: 'bold' }}>
          예측 결과
        </span>
      </h4>
      <p style={{ textAlign: 'center', fontSize: 20 }}>
        이 환자의 욕창 발생 위험도는 <strong>{data.risk_score.toFixed(1)}%</strong> 입니다.
      </p>
      {data.features.length > 0 && (
        <ul style={{ textAlign: 'center', listStyle: 'none', padding: 0, marginTop: 16 }}>
          {data.features.map((f) => (
            <li key={f.feature} style={{ fontSize: 16, marginBottom: 6 }}>
              <strong>{f.feature}</strong>이 위험 증가에 <strong>{f.contribution}%</strong> 기여하였습니다
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function Home() {
  const [form, setForm] = useState({
    feature1: '',
    feature2: '',
    feature3: '',
    feature4: '',
    feature5: '',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [openFaqs, setOpenFaqs] = useState<Set<number>>(new Set());

  const set = (key: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [key]: e.target.value }));

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data: ApiResponse = await res.json();
      setResult(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'unknown';
      setResult({ error: '예측 중 오류가 발생했습니다: ' + msg });
    } finally {
      setLoading(false);
    }
  };

  const toggleFaq = (i: number) => {
    setOpenFaqs((s) => {
      const n = new Set(s);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return n;
    });
  };

  return (
    <>
      <Head>
        <title>욕창 발생 위험도 계산기</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="container mt-5">
        <h2 className="text-center text-primary" style={{ fontWeight: 'bold' }}>
          욕창 발생 위험도 계산기
        </h2>
        <p className="text-center text-muted">데이터값을 입력한 후 예측하기 버튼을 클릭하세요.</p>

        <form className="p-4 bg-white shadow rounded" onSubmit={submit}>
          <div className="mb-3">
            <label className="form-label">의식수준 (RASS -5 ~ +4)</label>
            <input
              type="number"
              step="0.1"
              name="feature1"
              className="form-control"
              required
              value={form.feature1}
              onChange={set('feature1')}
            />
            <small className="text-muted">
              -5에서 +4 사이의 값을 입력하세요. +1~+4의 경우 + 입력 없이 숫자만 입력해도 됩니다.
            </small>
          </div>

          <div className="mb-3">
            <label className="form-label">
              가장 <strong>낮은</strong> 체온
            </label>
            <input
              type="number"
              step="0.1"
              name="feature2"
              className="form-control"
              required
              value={form.feature2}
              onChange={set('feature2')}
            />
            <small className="text-muted">
              지난 1일 동안 가장 낮은 체온을 소수점 첫째자리까지 입력하세요. 37.0도처럼 정수인 경우에는 37만 입력해도 됩니다.
            </small>
          </div>

          <div className="mb-3">
            <label className="form-label">
              가장 <strong>높은</strong> 체온
            </label>
            <input
              type="number"
              step="0.1"
              name="feature3"
              className="form-control"
              required
              value={form.feature3}
              onChange={set('feature3')}
            />
            <small className="text-muted">
              지난 1일 동안 가장 높은 체온을 소수점 첫째자리까지 입력하세요. 37.0도처럼 정수인 경우에는 37만 입력해도 됩니다.
            </small>
          </div>

          <div className="mb-3">
            <label className="form-label">하지근력(0 ~ 10)</label>
            <input
              type="number"
              step="0.1"
              name="feature4"
              className="form-control"
              required
              value={form.feature4}
              onChange={set('feature4')}
            />
            <small className="text-muted">
              통합임상관찰에서 양쪽 하지근력을 더한 0에서 10 사이의 값을 입력하세요.
            </small>
          </div>

          <div className="mb-3">
            <label className="form-label">실금횟수</label>
            <input
              type="number"
              step="0.1"
              name="feature5"
              className="form-control"
              required
              value={form.feature5}
              onChange={set('feature5')}
            />
            <small className="text-muted">1일 동안의 실금 횟수를 모두 더한 값을 입력하세요.</small>
          </div>

          <button type="submit" className="btn btn-primary w-100" disabled={loading}>
            예측하기
          </button>

          {loading && (
            <div className="text-center mt-3">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">계산 중...</span>
              </div>
              <p className="text-muted mt-2">계산 중입니다. 잠시만 기다려 주세요...</p>
            </div>
          )}

          {result && 'error' in result && (
            <div className="alert alert-danger mt-3" role="alert">
              {result.error}
            </div>
          )}
        </form>

        {result && !('error' in result) && <ResultCard data={result} />}

        <div className="usage-guide mt-4 p-4 bg-white shadow rounded">
          <h3 className="text-primary font-weight-bold" style={{ fontSize: 22, textAlign: 'left' }}>
            이용방법
          </h3>
          <ol className="pl-3" style={{ fontSize: 20 }}>
            <br />
            <li>
              값을 입력하고 <strong>예측하기</strong> 버튼을 누르면 욕창 위험도와 기여 요인이 자동으로 계산됩니다.
            </li>
            <li>
              어떤 값을 넣어야 하는지(예 : 어제 하지근력을 모두 더해서 평균을 구해야 하는지, 의식수준은 언제를 기준으로 해야하는지 등) 는 <strong>중환자간호팀 혹은 각 부서 WOC 의 지침</strong> 을 따라주세요.
            </li>
            <li>
              입력하는 값은 서버에 저장되지 않으며 다양한 값을 시뮬레이션해보면서 위험도가 어떻게 변화하는지 확인해보세요.
            </li>
          </ol>
          <div
            style={{
              textAlign: 'right',
              marginTop: 10,
              marginRight: 10,
              fontSize: 14,
              fontWeight: 'bold',
            }}
          >
            developed by. RN 김경란, RN 국서라
          </div>
        </div>

        <h3 className="text-center mt-5 text-primary">
          <strong>자주 묻는 질문 (FAQ)</strong>
        </h3>
        <div className="qa-container">
          {FAQS.map((item, i) => {
            const open = openFaqs.has(i);
            return (
              <div className="qa-item" key={i}>
                <div
                  className={`question ${open ? 'active' : ''}`}
                  onClick={() => toggleFaq(i)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && toggleFaq(i)}
                >
                  {item.q}
                  <span className={`icon ${open ? 'rotate' : ''}`}>{open ? '−' : '+'}</span>
                </div>
                <div className={`answer ${open ? 'show' : ''}`}>{item.a}</div>
              </div>
            );
          })}
        </div>
      </div>

      <ChatWidget />
    </>
  );
}
