# Serves flower_web.py: loads the trained model from MLflow (run outside this
# container, e.g. via `mlflow ui` on the host) and classifies uploaded flower photos.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flower_web.py config.yaml ./

EXPOSE 8000

# Point this at your MLflow server when it isn't reachable at config.yaml's default
# (http://127.0.0.1:5000 only works if this container shares the host's network),
# e.g.: docker run -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ...
CMD ["uvicorn", "flower_web:app", "--host", "0.0.0.0", "--port", "8000"]
