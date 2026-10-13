# Serves flower_web.py: downloads the trained ONNX model from the Hugging Face Hub
# (gdepaepe/dai6, see config.yaml) on startup and classifies uploaded flower photos.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/flower/flower_web.py src/flower/config.yaml ./

EXPOSE 8000

CMD ["uvicorn", "flower_web:app", "--host", "0.0.0.0", "--port", "8000"]
