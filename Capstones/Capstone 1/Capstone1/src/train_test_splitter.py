from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Parts_Per_Hour"


def split_data(
    X: pd.DataFrame, y: pd.Series = None, test_size: float = 0.2, random_state: int = 42
):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series (optional, for backward compatibility)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Handle backward compatibility
    if y is None:
        if TARGET_COLUMN not in X.columns:
            raise ValueError("Target column 'Parts_Per_Hour' not found in DataFrame.")
        y = X[TARGET_COLUMN]
        X = X.drop(columns=[TARGET_COLUMN])
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
