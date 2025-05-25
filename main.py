from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from surrealdb import Surreal, RecordID
from models.api import initialize_model, analyze_pcap
from datetime import datetime
from pathlib import Path
from fastapi import File

import uuid

model, processor = initialize_model()
db = Surreal("ws://localhost:5070")
db.use("main", "dns")
db.signin({"username": "root", "password": "root"})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.put("/predict")
async def predict(file: bytes = File(description="PCAP file to analyze")):
    uploads_dir = Path.cwd().joinpath("uploads")
    if not uploads_dir.exists():
        uploads_dir.mkdir(parents=True, exist_ok=True)

    pcap_path = uploads_dir.joinpath(f"{uuid.uuid4()}.pcap")
    pcap_path.write_bytes(file)
    results, report = analyze_pcap(model, processor, str(pcap_path))

    results = {
        "results": [result.model_dump() for result in results],
        "report": report.model_dump(),
        "created_at": datetime.now(),
    }
    data = db.create("reports", results)
    return {
        "message": "Prediction completed",
        "report_id": data["id"].id,
        "data": results,
    }


@app.get("/report/{report_id}")
async def get_report(report_id: str):
    report = db.query(
        "SELECT * FROM reports WHERE id = $id",
        {"id": RecordID("reports", report_id)},
    )
    if not report:
        return {"error": "Report not found"}
    return report[0]
