FROM python:3.12-slim

WORKDIR /data
COPY data/model_step7.pkl .
COPY data/model_threshold_step7.pkl .
COPY data/dataset_test.pkl.zip .
COPY data/human_friendly_dataset_test.pkl.zip .

WORKDIR /tmp
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /api
COPY api/api.py .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8084"]
