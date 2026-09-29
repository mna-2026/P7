FROM python:3.12-slim

WORKDIR /app
COPY api/requirements.txt .
COPY api/api.py .
RUN pip install -r requirements.txt

WORKDIR /data
COPY data/model_step7.pkl .
COPY data/model_threshold_step7.pkl .
COPY data/dataset_test.pkl.zip .
COPY data/human_friendly_dataset_test.pkl.zip .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8084"]