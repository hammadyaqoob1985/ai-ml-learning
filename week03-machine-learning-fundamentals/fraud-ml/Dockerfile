FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "fraud_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]