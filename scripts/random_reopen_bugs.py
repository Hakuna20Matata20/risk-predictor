# scripts/random_reopen_bugs.py

import os
import random
from dotenv import load_dotenv
from jira import JIRA

# 1) Підвантажуємо .env
load_dotenv()
JIRA_URL    = os.getenv("JIRA_URL")
JIRA_EMAIL  = os.getenv("JIRA_EMAIL")
JIRA_TOKEN  = os.getenv("JIRA_TOKEN")
PROJECT_KEY = os.getenv("PROJECT_KEY")

# 2) Підключаємося до Jira
jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_EMAIL, JIRA_TOKEN))

def random_reopen_walk(issue_key, max_steps=4, reopen_prob=0.5):
    """
    Для однієї задачі робить до max_steps випадкових переходів.
    Кожного разу з ймовірністю reopen_prob спеціально відправляє в статус Reopened (якщо є така опція).
    """
    # Завантажуємо поточну Issue зі changelog
    iss = jira.issue(issue_key, expand="changelog")
    for _ in range(random.randint(1, max_steps)):
        trans = jira.transitions(iss)
        if not trans:
            break

        # З 50% шансом обираємо саме перехід у Reopened
        if random.random() < reopen_prob:
            # знайдемо перехід, що веде в Reopened
            reopen_trans = [t for t in trans if t["name"].lower() == "reopened"]
            if reopen_trans:
                jira.transition_issue(iss, reopen_trans[0]["id"])
                print(f"{issue_key} → Reopened")
                # оновимо локальну копію
                iss = jira.issue(issue_key, expand="changelog")
                continue

        # Інакше — випадковий доступний перехід
        choice = random.choice(trans)
        jira.transition_issue(iss, choice["id"])
        print(f"{issue_key} → {choice['name']}")
        iss = jira.issue(issue_key, expand="changelog")

if __name__ == "__main__":
    # 3) Витягуємо всі Bug-закриття в проекті
    jql = f"project = {PROJECT_KEY} AND issuetype = Bug"
    bugs = jira.search_issues(jql, maxResults=100)

    for bug in bugs:
        # Для кожного з ймовірністю 0.5 робимо історію reopen
        random_reopen_walk(bug.key, max_steps=4, reopen_prob=0.5)
