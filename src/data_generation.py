# src/data_generation.py

import os
import numpy as np
import pandas as pd

def generate_synthetic_project_data(
    n_samples: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Генерує синтетичний датасет з такими колонками:
      - estimate_h     : оцінка задач у годинах
      - time_spent_h   : фактично витрачений час у годинах
      - changes_count  : скільки разів задачі переходили в статус Reopened
      - bug_count      : кількість багів в проєкті
      - budget         : фінансовий бюджет проєкту (умовні тис. $)
      - team_experience: середній досвід команди в роках
      - risk           : цільова змінна (0 = low risk, 1 = high risk)
    """
    rng = np.random.default_rng(random_state)

    # Згенеруємо фічі
    estimate_h     = rng.normal(loc=50, scale=20, size=n_samples).clip(min=1)
    time_spent_h   = (estimate_h * rng.uniform(0.8, 1.2, size=n_samples)).clip(min=0)
    changes_count  = rng.integers(0, 10, size=n_samples)
    bug_count      = rng.integers(0, 20, size=n_samples)
    budget         = rng.normal(loc=200, scale=50, size=n_samples).clip(min=10)
    team_experience= rng.normal(loc=3, scale=1.5, size=n_samples).clip(min=0)

    # Простий приклад розрахунку risk_score як середнього від нормованих фіч
    risk_score = (
        (estimate_h     / estimate_h.max())     * 0.2 +
        (time_spent_h   / time_spent_h.max())   * 0.2 +
        (changes_count  / changes_count.max())  * 0.2 +
        (bug_count      / bug_count.max())      * 0.2 +
        ((3 - team_experience) / 3)             * 0.2
    )
    risk = (risk_score > 0.5).astype(int)

    df = pd.DataFrame({
        "estimate_h":       estimate_h,
        "time_spent_h":     time_spent_h,
        "changes_count":    changes_count,
        "bug_count":        bug_count,
        "budget":           budget,
        "team_experience":  team_experience,
        "risk":             risk
    })

    return df

if __name__ == "__main__":
    # Генеруємо 1000 зразків і зберігаємо в data/synthetic_data.csv
    df = generate_synthetic_project_data(n_samples=1000, random_state=42)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_data.csv", index=False)
    print("✅ Synthetic data saved to data/synthetic_data.csv")
