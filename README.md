# E2E Wine Quality Prediction
Airflow와 MLflow로 구현한 엔드투엔드 와인 품질 분류 파이프라인 예제입니다.
데이터 수집 → 전처리 → 학습 → 평가 → 모델 등록 과정을 자동화합니다.

### 기술 스택
- Airflow: 워크플로 관리
- MLflow: 실험 기록 및 모델 레지스트리
- Postgres: 메타데이터 저장
- Python 3.12, scikit-learn, imbalanced-learn(SMOTE)

### 실행 방법
1) .env 작성
```
AIRFLOW_UID=1000
```
2) 컨테이너 실행
```
docker compose up -d
```
Airflow UI: http://localhost:8080
기본 계정: airflow / airflow
MLflow UI: http://localhost:5000

3) DAG 실행
Airflow UI에서 wine_quality_prediction DAG을 트리거합니다.

### 파이프라인 단계
1. 데이터 수집: UCI 와인 품질 데이터 다운로드
2. 유효성 검사: 컬럼/결측치 검증
3. 전처리: 타겟 변환, 데이터 분할, SMOTE
4. 학습: RandomForest 분류기, MLflow 기록
5. 평가: ROC AUC 등 지표 로깅
6. 모델 등록: 성능 비교 후 레지스트리 업데이트

### 참고
- MLflow 아티팩트 경로(/mlruns)를 Airflow 서비스에도 동일하게 마운트해야 로컬 파일 아티팩트를 읽을 수 있습니다.
- Alembic 리비전 충돌 방지를 위해 Airflow DB와 MLflow DB/USER를 분리하세요.
