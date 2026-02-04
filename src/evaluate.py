import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.load_data import load_processed_data, load_model, BASE_PATH
from sklearn.metrics import mean_squared_error, r2_score

log_dir = BASE_PATH / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "evaluate.log")
    ]
)
logger = logging.getLogger()

def run_evaluation(plot=True):
    model = load_model()
    data = load_processed_data()
    X_test = data["X_test"]
    y_test = data["y_test"]**2 # converting back to it's original scale

    y_pred = model.predict(X_test)
    y_pred = y_pred**2 # converting back to it's original scale

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {'MSE': round(mse, 2), 'RMSE': round(rmse, 2), 'R2': r2}
    logger.info(f"Evaluation Metrics: {metrics}")

    if plot:
        residuals = y_test - y_pred
        plt.figure(figsize=(12,5))
        
        # Residual plot
        plt.subplot(1,2,1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Actual vs Predicted
        plt.subplot(1,2,2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()

    model_name = model.named_steps["model"].__class__.__name__
    return pd.DataFrame([metrics], index=[model_name])


if __name__ == "__main__":
    run_evaluation()