FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model.tflite
ENV STATS_PATH=/app/training_stats.npy
ENV MQTT_HOST=host.docker.internal
ENV MQTT_PORT=1883
ENV TRUCK_ID=TRUCK-001
ENV MQTT_QOS=1

WORKDIR /app

COPY inference/requirements.txt ./requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip \
    && python -m pip install \
    --no-cache-dir \
    -r requirements.txt

COPY data_pipeline/preprocessing.py ./preprocessing.py
COPY data_pipeline/training_stats.npy ./training_stats.npy
COPY inference/inference_service.py ./inference_service.py

# Assignment OTA/cache requirement:
# Every pip installation layer appears before this model-copy layer.
COPY inference/model.tflite ./model.tflite

CMD ["python", "inference_service.py"]
