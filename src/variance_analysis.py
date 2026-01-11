import pandas as pd
import numpy as np

def compute_relative_variance_cv(df, anchor_col="Overall Score"):
    import numpy as np

    anchor_idx = df.columns.get_loc(anchor_col)

    numeric_df = (
        df.iloc[:, anchor_idx + 1 :]
        .select_dtypes(include=np.number)
        .dropna()
    )

    cv = numeric_df.std() / numeric_df.mean().abs()
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()

    return cv.sort_values(ascending=False)