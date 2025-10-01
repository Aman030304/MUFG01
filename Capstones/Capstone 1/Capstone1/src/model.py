import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def _log_top_coefficients(model_name: str, feature_names: List[str], coefficients: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    top = coef_df.head(top_k)
    logging.info(f"Top {top_k} features by |coefficient| for {model_name}:")
    for _, row in top.iterrows():
        logging.info(f"  {row['feature']}: {row['coefficient']:.6f}")
    return coef_df


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Tuple[str, ...] = ("linear", "ridge", "lasso"),
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 0.01,
    save_dir: Path | None = None,
) -> Dict[str, object]:
    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[1] / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    trained: Dict[str, object] = {}

    feature_names = list(X_train.columns)

    if "linear" in models:
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        trained["linear"] = lin
        coef = getattr(lin, "coef_", np.zeros(len(feature_names)))
        _log_top_coefficients("LinearRegression", feature_names, coef)
        joblib.dump(lin, save_dir / "linear_regression.pkl")
        logging.info(f"Saved Linear Regression model to {save_dir / 'linear_regression.pkl'}")

    if "ridge" in models:
        ridge = Ridge(alpha=ridge_alpha, random_state=42)
        ridge.fit(X_train, y_train)
        trained["ridge"] = ridge
        coef = getattr(ridge, "coef_", np.zeros(len(feature_names)))
        _log_top_coefficients("Ridge", feature_names, coef)
        joblib.dump(ridge, save_dir / "ridge_regression.pkl")
        logging.info(f"Saved Ridge model to {save_dir / 'ridge_regression.pkl'}")

    if "lasso" in models:
        lasso = Lasso(alpha=lasso_alpha, random_state=42, max_iter=10000)
        lasso.fit(X_train, y_train)
        trained["lasso"] = lasso
        coef = getattr(lasso, "coef_", np.zeros(len(feature_names)))
        _log_top_coefficients("Lasso", feature_names, coef)
        joblib.dump(lasso, save_dir / "lasso_regression.pkl")
        logging.info(f"Saved Lasso model to {save_dir / 'lasso_regression.pkl'}")

    return trained


def compare_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path | None = None,
    scaler=None,
    encoder=None,
) -> pd.DataFrame:
    """
    Evaluate all trained models and return comparison dataframe.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        save_dir: Directory to save best model
        
    Returns:
        DataFrame with model comparison results
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[1] / "models"
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'R2': r2
        })
        
        logging.info(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('R2', ascending=False)
    
    # Save best performing model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    # Save best model with descriptive name
    best_model_path = save_dir / f"best_{best_model_name}_regression.pkl"
    joblib.dump(best_model, best_model_path)
    logging.info(f"Saved best model ({best_model_name}) to {best_model_path}")
    
    # Optionally save transformers used during training for inference
    if scaler is not None:
        scaler_path = save_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved fitted scaler to {scaler_path}")
    if encoder is not None:
        encoder_path = save_dir / "encoder.pkl"
        joblib.dump(encoder, encoder_path)
        logging.info(f"Saved feature encoder to {encoder_path}")
    
    # Save comparison results
    comparison_path = save_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logging.info(f"Saved model comparison to {comparison_path}")
    
    return comparison_df