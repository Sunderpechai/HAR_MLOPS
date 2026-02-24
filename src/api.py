from fastapi import FastAPI
import mlflow.pyfunc
import logging
import time
from prometheus_client import Counter, Histogram, start_http_server

app = FastAPI()

# Load Production model
model = mlflow.pyfunc.load_model("models:/HAR_Model/Production")

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO
)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total Prediction Requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

start_http_server(8001)

@app.post("/predict")
def predict(features: list):

    REQUEST_COUNT.inc()
    start_time = time.time()

    prediction = model.predict([features])

    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)

    logging.info(f"Prediction: {prediction[0]} | Latency: {latency}")

    return {"predicted_activity": prediction[0]}
