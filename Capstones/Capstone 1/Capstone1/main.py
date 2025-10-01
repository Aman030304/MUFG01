import logging

from src.data_loader import load_data
from src.preprocessing import preprocess, scale_features, build_encoder_from_df
from src.train_test_splitter import split_data
from src.model import train_models, compare_models
from src.evaluate import evaluate_model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Loading dataset...")
    raw_df = load_data()

    logging.info("Preprocessing data...")
    df = preprocess(raw_df)
    
    logging.info("Building feature encoder from preprocessed training frame...")
    encoder = build_encoder_from_df(df)

    logging.info("Scaling features...")
    X_scaled, y, scaler = scale_features(df)

    logging.info("Splitting train/test...")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    logging.info("Training models (Linear, Ridge, Lasso)...")
    models = train_models(X_train, y_train)

    logging.info("Comparing models...")
    comparison_df = compare_models(models, X_test, y_test, scaler=scaler, encoder=encoder)
    
    print("\n=== MODEL COMPARISON ===")
    print(comparison_df.to_string(index=False))

    logging.info("Evaluating best model with visualization...")
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
    mse, r2 = evaluate_model(
        best_model,
        X_test,
        y_test,
        plot=False,
        title=f"{best_model_name.title()} Regression - Predicted vs Actual"
    )
    
    print(f"\nBest model performance:")
    print(f"  Model: {best_model_name}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    logging.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
