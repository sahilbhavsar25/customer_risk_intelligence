PY ?= python

.PHONY: help data clean-data validate features train pipeline \
        db-load ingest eval-dataset eval rag demo test api all

help:
	@echo "make pipeline   generate -> clean -> validate -> features -> train -> SHAP"
	@echo "make db-load    sync processed data into PostgreSQL"
	@echo "make ingest     embed new/changed documents into Qdrant"
	@echo "make eval       build RAG eval set and run the evaluation"
	@echo "make demo       idempotency / incremental / failure / schema demos"
	@echo "make test       run pytest"
	@echo "make api        run the API on :8000"
	@echo "make all        everything above, in order"

data:
	$(PY) -m src.data_generation.generate_all

clean-data:
	$(PY) -m src.data_engineering.profiler
	$(PY) -m src.data_engineering.validator
	$(PY) -m src.data_engineering.cleaner
	$(PY) -m src.data_engineering.processed_validator

features:
	$(PY) -m src.features.feature_engineer
	$(PY) -m src.ml.prepare_dataset

train:
	$(PY) -m src.ml.train_models
	$(PY) -m src.ml.explain_model

pipeline: data clean-data features train

db-load:
	$(PY) -m src.database.load_processed

ingest:
	$(PY) -m src.rag.ingest_documents

eval-dataset:
	$(PY) -m src.rag.build_evaluation_dataset

eval: eval-dataset
	$(PY) -m src.rag.evaluate_rag

rag: ingest eval

demo:
	$(PY) -m src.data_engineering.pipeline_demo

test:
	$(PY) -m pytest

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

all: pipeline db-load ingest eval demo test
