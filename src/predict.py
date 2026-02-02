from src.load_data import load_model

def predict(X):
    """
    Run model prediction.
    
    Args:
        model: trained model
        X: features for prediction
    
    Returns:
        y_pred
    """
    model = load_model()
    return model.predict(X)