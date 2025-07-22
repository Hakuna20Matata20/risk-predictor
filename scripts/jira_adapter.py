# scripts/jira_adapter.py

import os
from dotenv import load_dotenv
import pandas as pd
from jira import JIRA

# 1) Завантажуємо змінні з .env
load_dotenv()  # шукає файл .env у корені

JIRA_URL    = os.getenv("JIRA_URL")
JIRA_EMAIL  = os.getenv("JIRA_EMAIL")
JIRA_TOKEN  = os.getenv("JIRA_TOKEN")
PROJECT_KEY = os.getenv("PROJECT_KEY")

# 2) Підключаємося до Jira
jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_EMAIL, JIRA_TOKEN))

def fetch_project_metrics(project_key: str, max_results: int = 1000) -> dict:
    """
    Завантажує метрики по всіх Issue та по Bug-типам окремо.
    Повертає словник:
      - project_name  : ключ проекту
      - tasks_count   : загальна кількість Issue
      - bug_count     : кількість типу Bug
      - changes_count : сумарна кількість переходів у Reopened
      - estimate_h    : сума Original Estimate у годинах
      - time_spent_h  : сума Time Spent у годинах
    """
    # всі Issue з changelog
    all_issues = jira.search_issues(
        f"project = {project_key}",
        maxResults=max_results,
        expand="changelog"
    )

    # окремий запит для Bug-ів
    bug_issues = jira.search_issues(
        f"project = {project_key} AND issuetype = Bug",
        maxResults=max_results
    )

    tasks_count  = len(all_issues)
    bug_count    = len(bug_issues)
    reopened_cnt = 0
    total_est    = 0
    total_spent  = 0

    for iss in all_issues:
        # лічимо Reopened transitions
        for hist in iss.changelog.histories:
            for itm in hist.items:
                if itm.field == "status" and itm.toString.lower() == "reopened":
                    reopened_cnt += 1

        # збираємо Original Estimate та Time Spent (в секундах)
        est = iss.fields.timeoriginalestimate or 0
        spent = iss.fields.timespent or 0
        total_est   += est
        total_spent += spent

    # конвертуємо в години
    estimate_h   = total_est   / 3600.0
    time_spent_h = total_spent / 3600.0

    return {
        "project_name":   project_key,
        "tasks_count":    tasks_count,
        "bug_count":      bug_count,
        "changes_count":  reopened_cnt,
        "estimate_h":     estimate_h,
        "time_spent_h":   time_spent_h
    }

if __name__ == "__main__":
    # 3) Викликаємо збір метрик і зберігаємо в CSV
    metrics = fetch_project_metrics(PROJECT_KEY)
    df = pd.DataFrame([metrics])

    os.makedirs("data", exist_ok=True)
    out_path = "data/jira_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Jira metrics exported to {out_path} (n_projects=1)")

