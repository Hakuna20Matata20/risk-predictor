# scripts/update_timetracking.py
import os, random
from dotenv import load_dotenv
from jira import JIRA

load_dotenv()
jira = JIRA(os.getenv("JIRA_URL"), basic_auth=(os.getenv("JIRA_EMAIL"), os.getenv("JIRA_TOKEN")))
PROJECT = os.getenv("PROJECT_KEY")

# Завантажуємо всі задачі (Tasks і Bugs)
issues = jira.search_issues(f"project={PROJECT}", maxResults=500)

for iss in issues:
    # Випадковий estimate між 1h та 16h
    est_hours = random.randint(1, 16)
    # Time spent — 50–120% від estimate
    spent_hours = int(est_hours * random.uniform(0.5, 1.2))

    jira.issue_update(
        iss.key,
        fields={
            'timetracking': {
                'originalEstimate': f"{est_hours}h",
                'timeSpent':        f"{spent_hours}h"
            }
        }
    )
    print(f"{iss.key}: estimate={est_hours}h, spent={spent_hours}h")
