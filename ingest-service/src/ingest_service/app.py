from fastapi import FastAPI
from ingest_service.producer import FrameProducer

app = FastAPI()

producer = FrameProducer()

@app.post("/start")  # Prefer POST for starting actions
async def start_ingest():
    producer.start_stream()
    return {"message": "Ingestion started"}
