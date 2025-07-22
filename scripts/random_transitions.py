import os, random
from dotenv import load_dotenv
from jira import JIRA

load_dotenv()
jira = JIRA(os.getenv("JIRA_URL"), basic_auth=(os.getenv("JIRA_EMAIL"), os.getenv("JIRA_TOKEN")))
PROJECT = os.getenv("PROJECT_KEY")

def random_walk(key, steps=5):
    for _ in range(steps):
        iss = jira.issue(key)
        trans = jira.transitions(iss)
        if not trans: break
        choice = random.choice(trans)
        jira.transition_issue(iss, choice["id"])
        print(f"{key} → {choice['name']}")

if __name__ == "__main__":
    issues = jira.search_issues(f"project={PROJECT}", maxResults=50)
    for i in issues:
        random_walk(i.key, steps=random.randint(1,4))
