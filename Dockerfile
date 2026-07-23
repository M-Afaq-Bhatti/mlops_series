FROM python:3.14-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Copy only the dependency file first — Docker caches this layer,
# so re-builds are fast as long as dependencies haven't changed
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Now copy the actual application code and trained model
COPY app/ ./app
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]