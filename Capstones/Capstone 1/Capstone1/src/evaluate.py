from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, plot: bool = False, title: str | None = None) -> Tuple[float, float]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")

    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
        
        # Add diagonal line for perfect prediction
        min_val = float(min(np.min(y_test), np.min(y_pred)))
        max_val = float(max(np.max(y_test), np.max(y_pred)))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add labels and title
        plt.xlabel("Actual Parts Per Hour", fontsize=12)
        plt.ylabel("Predicted Parts Per Hour", fontsize=12)
        plt.title(title or "Predicted vs Actual Values", fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return mse, r2
