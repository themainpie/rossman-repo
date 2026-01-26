import pandas as pd
from pathlib import Path


def load_train():
    """load train CSV"""
    base_path = Path(__file__).resolve().parents[1] / "data/raw"
    train = pd.read_csv(base_path / "train.csv")

    return train



def load_test():
    """load test CSV"""
    base_path = Path(__file__).resolve().parents[1] / "data/raw"
    test = pd.read_csv(base_path / "test.csv")

    return test



def load_store():
    """load store CSV"""
    base_path = Path(__file__).resolve().parents[1] / "data/raw"
    store = pd.read_csv(base_path / "store.csv")

    return store