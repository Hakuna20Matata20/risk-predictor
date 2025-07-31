```python
# scripts/jira_adapter.py

import os
from dotenv import load_dotenv
import pandas as pd
from jira import JIRA

# Load environment variables
load_dotenv()
JIRA_URL    = os.getenv("JIRA_URL")
JIRA_EMAIL  = os.getenv("JIRA_EMAIL")
JIRA_TOKEN  = os.getenv("JIRA_TOKEN")
PROJECT_KEY = os.getenv("PROJECT_KEY")

# Custom field IDs
CF_ESTIMATE_H = "customfield_10046"  # Estimate Hours
CF_SPENT_H    = "customfield_10047"  # Time Spent Hours


def get_jira_client() -> JIRA:
    """
    Create and return a JIRA client. Lazy initialization to avoid blocking at import.
    """
    return JIRA(
        server=JIRA_URL,
        basic_auth=(JIRA_EMAIL, JIRA_TOKEN),
        options={"rest_api_timeout": 20}
    )


def fetch_project_metrics(project_key: str, max_results: int = 1000) -> pd.DataFrame:
    """
    Fetch project metrics:
      - tasks_count
      - bug_count
      - changes_count (reopened)
      - estimate_h (sum of Estimate Hours)
      - time_spent_h (sum of Time Spent Hours)
    Returns a single-row DataFrame.
    """
    # Lazy initialize JIRA client
    jira = get_jira_client()

    # Fetch all issues with changelog for reopen count
    all_issues = jira.search_issues(
        f"project = {project_key}",
        maxResults=max_results,
        expand="changelog"
    )
    # Fetch only bugs
    bug_issues = jira.search_issues(
        f"project = {project_key} AND issuetype = Bug",
        maxResults=max_results
    )

    tasks_count  = len(all_issues)
    bug_count    = len(bug_issues)
    reopened_cnt = 0
    total_est    = 0.0
    total_spent  = 0.0

    for iss in all_issues:
        # Count reopened transitions
        for history in iss.changelog.histories:
            for item in history.items:
                if item.field == "status" and item.toString.lower() == "reopened":
                    reopened_cnt += 1
        # Read custom fields
        fields = iss.fields
        est_val   = getattr(fields, CF_ESTIMATE_H, 0) or 0
        spent_val = getattr(fields, CF_SPENT_H,    0) or 0
        total_est   += float(est_val)
        total_spent += float(spent_val)

    data = {
        "project_name":   project_key,
        "tasks_count":    tasks_count,
        "bug_count":      bug_count,
        "changes_count":  reopened_cnt,
        "estimate_h":     total_est,
        "time_spent_h":   total_spent
    }
    return pd.DataFrame([data])
```
