import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from feature_engineering import preprocess_features


def train_and_save_model(
        data_path: str = "data/synthetic_data.csv",
        model_path: str = "models/xgb_model.joblib"
):
    # 1) Завантажуємо дані
    df = pd.read_csv(data_path)

    # 2) Інженеримо фічі
    X, y = preprocess_features(df)

    # 3) Розбиваємо на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4) Створюємо модель
    model = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    # 5) Тренуємо
    model.fit(X_train, y_train)

    # 6) Створюємо папку models/, якщо нема
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 7) Зберігаємо модель
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
