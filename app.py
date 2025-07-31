import streamlit as st
import pandas as pd
import joblib
from scripts.jira_adapter import fetch_project_metrics
from src.feature_engineering import preprocess_features

model = joblib.load("models/xgb_model.joblib")

st.title("📊 Project Risk Predictor")

# 1) отримати список ключів (ви можете hard-код або зробити fetch через jira.projects())
project_keys = ["TEST","DEMO","RISK"]
sel = st.multiselect("Select Jira project(s)", project_keys)

rows = []
if sel:
    for p in sel:
        m = fetch_project_metrics(p)
        st.markdown(f"### {p}")
        st.write(f"- Tasks: {m['tasks_count']}")
        st.write(f"- Reopens: {m['changes_count']}")
        st.write(f"- Bugs: {m['bug_count']}")
        st.write(f"- Estimate (h): {m['estimate_h']:.1f}")
        st.write(f"- Time spent (h): {m['time_spent_h']:.1f}")

        te = st.number_input(f"Team experience for {p}", 0.0, 20.0, 3.0, key=f"te_{p}")
        bud = st.number_input(f"Project budget for {p}", 0.0, 10000.0, 100.0, key=f"bud_{p}")

        rows.append({
            "estimate_h": m["estimate_h"],
            "time_spent_h": m["time_spent_h"],
            "changes_count": m["changes_count"],
            "bug_count": m["bug_count"],
            "budget": bud,
            "team_experience": te
        })

    df_in = pd.DataFrame(rows)
    X, _ = preprocess_features(df_in)
    preds = model.predict(X)
    df_in["predicted_risk"] = preds

    st.subheader("Predictions")
    st.dataframe(df_in)
    csv = df_in.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "predictions.csv")
