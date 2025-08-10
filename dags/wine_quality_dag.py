from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator


def ingest_data(**kwargs):
    """
    UCI 저장소에서 와인 품질 데이터를 다운로드하고,
    파일 경로를 XCom에 저장합니다.
    """
    from pathlib import Path
    import pandas as pd

    # 데이터셋 URL
    print("Ingesting Data...")
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    # 데이터 저장 경로 설정
    data_dir = Path("/opt/airflow/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_data_path = data_dir / "winequality-red.csv"

    # 데이터 다운로드 및 저장
    df = pd.read_csv(URL, sep=";")
    df.to_csv(raw_data_path, index=False)

    # 다음 태스크를 위해 파일 경로를 XCom에 push
    ti = kwargs["ti"]
    ti.xcom_push(key="raw_data_path", value=str(raw_data_path))
    print(f"Data ingested and saved to {raw_data_path}")


def validate_data(**kwargs):
    """
    XCom에서 데이터 경로를 가져와 데이터의 유효성을 검사합니다.
    """
    import pandas as pd

    print("Validating Data...")
    ti = kwargs["ti"]
    raw_data_path = ti.xcom_pull(key="raw_data_path", task_ids="ingest_data")

    df = pd.read_csv(raw_data_path)

    # 1. 컬럼 존재 여부 확인
    expected_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("Dataset is missing expected columns.")

    # 2. Null 값 확인 (이 데이터셋은 null이 없다고 알려져 있음)
    if df.isnull().sum().sum() > 0:
        raise ValueError("Unexpected null values found in the dataset.")

    print("Data validation successful.")
    # 유효성 검사를 통과한 데이터 경로를 다시 push (또는 동일 경로 사용)
    ti.xcom_push(key="validated_data_path", value=raw_data_path)


def preprocess_data(**kwargs):
    """
    데이터를 전처리하고, 피처를 엔지니어링하며, SMOTE를 적용합니다.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from pathlib import Path

    print("Preprocessing Data...")

    ti = kwargs["ti"]
    validated_data_path = ti.xcom_pull(
        key="validated_data_path", task_ids="validate_data"
    )

    df = pd.read_csv(validated_data_path)

    # 1. 이진 타겟 변수 생성
    df["good_quality"] = [1 if x > 6 else 0 for x in df["quality"]]
    df = df.drop("quality", axis=1)

    # 2. 데이터 분할
    X = df.drop("good_quality", axis=1)
    y = df["good_quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. SMOTE 적용 (학습 데이터에만)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("Original training set shape:", y_train.value_counts())
    print("Resampled training set shape:", y_train_smote.value_counts())

    # 4. 전처리된 데이터 저장
    data_dir = Path("/opt/airflow/data")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    pd.concat([X_train_smote, y_train_smote], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    # 경로들을 XCom에 push
    ti.xcom_push(key="train_path", value=str(train_path))
    ti.xcom_push(key="test_path", value=str(test_path))


def train_model(**kwargs):
    """
    모델을 학습하고 MLflow에 실험 정보를 기록합니다.
    """
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    print("Training Model...")

    ti = kwargs["ti"]
    train_path = ti.xcom_pull(key="train_path", task_ids="preprocess_data")

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop("good_quality", axis=1)
    y_train = train_df["good_quality"]

    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("wine_quality_prediction")

    with mlflow.start_run() as run:
        # MLflow 자동 로깅 활성화
        mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

        # 모델 정의 및 학습
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        mlflow.log_metric("training_accuracy", train_accuracy)

        # run_id를 XCom에 push
        run_id = run.info.run_id
        ti.xcom_push(key="mlflow_run_id", value=run_id)
        print(f"Model trained. Run ID: {run_id}")


def evaluate_model(**kwargs):
    """
    테스트 데이터로 모델을 평가하고, 결과를 MLflow에 기록합니다.
    """
    import pandas as pd
    import mlflow
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    print("Evaluating Model...")

    ti = kwargs["ti"]
    test_path = ti.xcom_pull(key="test_path", task_ids="preprocess_data")
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train_model")

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop("good_quality", axis=1)
    y_test = test_df["good_quality"]

    mlflow.set_tracking_uri("http://mlflow-server:5000")

    # 학습된 모델 로드
    logged_model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

    predictions = loaded_model.predict(X_test)

    # 성능 지표 계산
    roc_auc = roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # 기존 run에 테스트 메트릭을 추가로 로깅
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "test_roc_auc": roc_auc,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
            }
        )

    print(f"Model evaluated. Test ROC AUC: {roc_auc}")
    ti.xcom_push(key="test_roc_auc", value=roc_auc)


def register_model(**kwargs):
    """
    모델 성능을 비교하여 조건부로 모델 레지스트리에 등록/승격합니다.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    print("Registring Model...")
    ti = kwargs["ti"]
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train_model")
    new_model_roc_auc = ti.xcom_pull(key="test_roc_auc", task_ids="evaluate_model")

    model_name = "wine-quality-classifier"
    model_uri = f"runs:/{run_id}/model"

    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = MlflowClient()

    try:
        # 현재 Production 단계의 모델 버전 정보 가져오기
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            production_model = latest_versions
            production_run_id = production_model.run_id
            production_roc_auc = client.get_run(production_run_id).data.metrics[
                "test_roc_auc"
            ]

            print(f"Current Production Model ROC AUC: {production_roc_auc}")
            print(f"New Model ROC AUC: {new_model_roc_auc}")

            # 성능 비교
            if new_model_roc_auc > production_roc_auc:
                print("New model is better. Promoting to Production.")
                # 새 모델 등록 및 Production으로 전환
                new_model_version = mlflow.register_model(model_uri, model_name).version
                client.transition_model_version_stage(
                    name=model_name,
                    version=new_model_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
            else:
                print(
                    "New model is not better than the production model. No action taken."
                )
        else:
            # 운영 모델이 없는 경우, 새 모델을 바로 Production으로 등록
            print("No production model found. Registering new model as Production.")
            new_model_version = mlflow.register_model(model_uri, model_name).version
            client.transition_model_version_stage(
                name=model_name, version=new_model_version, stage="Production"
            )
    except mlflow.exceptions.RestException:
        # 모델 자체가 등록되지 않은 경우
        print(f"Model '{model_name}' not found. Registering new model as Production.")
        new_model_version = mlflow.register_model(model_uri, model_name).version
        client.transition_model_version_stage(
            name=model_name, version=new_model_version, stage="Production"
        )


with DAG(
    dag_id="wine_quality_prediction",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["mlops", "classification", "portfolio"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    # 태스크 의존성 설정
    (
        ingest_task
        >> validate_task
        >> preprocess_task
        >> train_model_task
        >> evaluate_model_task
        >> register_model_task
    )
