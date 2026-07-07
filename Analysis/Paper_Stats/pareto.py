import numpy as np
import pandas as pd

def pareto_minimise(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:

    points = df[[x_col, y_col]].to_numpy()
    is_pareto = np.ones(len(points), dtype=bool)

    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    is_pareto[i] = False
                    break

    front = df.loc[is_pareto].copy()
    return front.sort_values([x_col, y_col]).reset_index(drop=True)
