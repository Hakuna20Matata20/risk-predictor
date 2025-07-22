import os
from dotenv import load_dotenv
import pandas as pd
from jira import JIRA

load_dotenv()
jira = JIRA(os.getenv("JIRA_URL"), basic_auth=(os.getenv("JIRA_EMAIL"), os.getenv("JIRA_TOKEN")))
PROJECT = os.getenv("PROJECT_KEY")

def fetch_project_metrics(project_key, max_results=1000):
    issues = jira.search_issues(f"project={project_key}", maxResults=max_results, expand="changelog")
    tasks = len(issues)
    reopened = 0
    bugs = 0
    total_est = 0
    total_spent = 0

    for iss in issues:
        for h in iss.changelog.histories:
            for itm in h.items:
                if itm.field=="status" and itm.toString.lower()=="reopened":
                    reopened += 1
        if iss.fields.issuetype.name.lower()=="bug":
            bugs += 1
        est = iss.fields.timeoriginalestimate or 0
        spent = iss.fields.timespent or 0
        total_est   += est
        total_spent += spent

    return {
        "project_name":  project_key,
        "tasks_count":   tasks,
        "changes_count": reopened,
        "bug_count":     bugs,
        "estimate_h":    total_est/3600,
        "time_spent_h":  total_spent/3600
    }

if __name__ == "__main__":
    m = fetch_project_metrics(PROJECT)
    df = pd.DataFrame([m])
    df.to_csv("data/jira_metrics.csv", index=False)
    print("✅ data/jira_metrics.csv created")
