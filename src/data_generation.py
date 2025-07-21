import numpy as np
import pandas as pd

def generate_synthetic_project_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Генерує DataFrame із такими стовпцями:
      - project_size: категорія розміру проєкту ('small', 'medium', 'large')
      - team_experience: середній досвід команди (роки)
      - budget: бюджет (тис. доларів)
      - tasks_count: кількість завдань
      - changes_count: кількість змін вимог
      - risk: цільова змінна (0 — low risk, 1 — high risk)
    """
    rng = np.random.default_rng(random_state)

    # 1) Генеруємо числові фічі
    team_experience = rng.normal(loc=3, scale=1.5, size=n_samples).clip(min=0)
    budget = rng.normal(loc=200, scale=50, size=n_samples).clip(min=50)
    tasks_count = rng.integers(20, 200, size=n_samples)
    changes_count = rng.poisson(lam=5, size=n_samples)

    # 2) Категоризуємо проєкт за розміром по бюджету
    project_size = pd.cut(
        budget,
        bins=[0, 100, 300, np.inf],
        labels=["small", "medium", "large"]
    )

    # 3) Обчислюємо простий risk_score та бінаризуємо
    risk_score = (
        (tasks_count / tasks_count.max()) * 0.3 +
        (changes_count / changes_count.max()) * 0.4 +
        ((3 - team_experience) / 3) * 0.3
    )
    risk = (risk_score > 0.5).astype(int)

    # 4) Збираємо у DataFrame
    df = pd.DataFrame({
        "project_size": project_size,
        "team_experience": team_experience,
        "budget": budget,
        "tasks_count": tasks_count,
        "changes_count": changes_count,
        "risk": risk
    })

    return df

if __name__ == "__main__":
    # Створюємо 1000 зразків і зберігаємо їх у data/synthetic_data.csv
    df = generate_synthetic_project_data(n_samples=1000, random_state=42)
    output_path = "../data/synthetic_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")
