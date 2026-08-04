FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy only the dependency file first, and only install CORE (serving)
# dependencies - NOT the [pipeline] extra (mlflow, prefect, dvc), which
# belongs on the training side, not in the serving image.
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Now copy the actual application code and trained model.
COPY app/ ./app
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
