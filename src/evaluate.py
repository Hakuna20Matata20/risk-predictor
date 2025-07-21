import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from feature_engineering import preprocess_features


def evaluate_model(
        data_path: str = "data/synthetic_data.csv",
        model_path: str = "models/xgb_model.joblib"
):
    # 1) Завантажуємо дані
    df = pd.read_csv(data_path)
    X, y = preprocess_features(df)

    # 2) Завантажуємо модель
    model = joblib.load(model_path)

    # 3) Робимо передбачення
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # 4) Виводимо звіт
    print("=== Classification Report ===")
    print(classification_report(y, y_pred))
    print("=== ROC AUC Score ===")
    print(roc_auc_score(y, y_proba))


if __name__ == "__main__":
    evaluate_model()
