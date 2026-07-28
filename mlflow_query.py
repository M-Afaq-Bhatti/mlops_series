import mlflow

runs = mlflow.search_runs(
    experiment_names=["student-score-predictor"],
    order_by=["metrics.mae ASC"],
)
print(runs[["run_id", "params.n_estimators", "params.max_depth", "metrics.mae", "metrics.r2"]])
