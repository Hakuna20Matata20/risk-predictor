import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_features(df: pd.DataFrame):
    """
    Приймає DataFrame з колонками:
      project_size (категорія),
      team_experience,
      budget,
      tasks_count,
      changes_count,
      risk (цільова)
    Повертає:
      X: DataFrame з числовими фічами та one-hot кодуванням project_size
      y: Series з мітками risk
    """
    # Створюємо копію, щоб не ламати оригінал
    df = df.copy()

    # 1) One-hot кодування для project_size
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    size_encoded = ohe.fit_transform(df[['project_size']])
    # Отримаємо список назв нових колонок
    size_cols = [f"size_{cat}" for cat in ohe.categories_[0]]
    df_ohe = pd.DataFrame(size_encoded, columns=size_cols, index=df.index)

    # 2) Вибираємо числові колонки
    numeric_cols = ['team_experience', 'budget', 'tasks_count', 'changes_count']
    df_num = df[numeric_cols]

    # 3) Об’єднуємо все разом
    X = pd.concat([df_num, df_ohe], axis=1)

    # 4) Цільова змінна
    y = df['risk']

    return X, y
