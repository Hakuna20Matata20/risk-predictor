import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(df: pd.DataFrame):
    df = df.copy()
    # тільки числові фічі:
    cols = ["estimate_h","time_spent_h","changes_count","bug_count","budget","team_experience"]
    X = df[cols]
    # можна масштабувати:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=cols), df.get("risk", None)
