import json

import joblib
import pandas as pd
import pytest

from src.ml.train_models import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture(scope="module")
def selection():
    with open("models/model_selection.json", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def artifacts(selection):
    model = joblib.load(f"models/{selection['selected_model']}.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    return model, preprocessor


def test_selection_file(selection):
    assert selection["selected_model"] in {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert selection["model_version"]


def test_model_loads(artifacts):
    model, preprocessor = artifacts
    assert hasattr(model, "predict_proba")
    assert hasattr(preprocessor, "transform")


def test_probabilities_between_0_and_1(artifacts):
    model, preprocessor = artifacts
    test = pd.read_csv("data/processed/ml/splits/test.csv")
    X = preprocessor.transform(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    probabilities = model.predict_proba(X)[:, 1]

    assert len(probabilities) == len(test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_beats_baseline():
    comparison = pd.read_csv("data/processed/ml/model_comparison.csv").set_index("model")
    baseline = comparison.loc["dummy_baseline", "pr_auc"]
    best = comparison.drop(index="dummy_baseline")["pr_auc"].max()
    assert best > baseline
