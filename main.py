from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from surrealdb import Surreal, RecordID
from models.api import initialize_model, analyze_pcap
from datetime import datetime
from pathlib import Path
from fastapi import File
from fastapi.responses import FileResponse

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

    filename = str(uuid.uuid4())
    pcap_path = uploads_dir.joinpath(f"{filename}.pcap")
    pcap_path.write_bytes(file)
    results, report = analyze_pcap(model, processor, str(pcap_path))

    report_data = {
        "report": report.model_dump(),
        "created_at": datetime.now(),
    }

    # Save the results as csv
    csv_path = uploads_dir.joinpath(f"{filename}.csv")
    with csv_path.open("w") as f:
        f.write("timestamp,query,prediction,confidence,confidence_percent\n")
        for result in results:
            f.write(
                f"{result.timestamp},{result.query},{result.prediction},"
                f"{result.confidence},{result.confidence_percent}\n"
            )
    report_data["csv_path"] = str(csv_path)
    report_data["pcap_path"] = str(pcap_path)

    data = db.create("reports", report_data)
    return {
        "message": "Prediction completed",
        "report_id": data["id"].id,
        "data": report_data,
    }


@app.get("/report/{report_id}")
async def get_report(report_id: str):
    report = db.query(
        "SELECT * FROM reports WHERE id = $id",
        {"id": RecordID("reports", report_id)},
    )
    if not report:
        return {"message": "Report not found"}
    report = report[0]
    report["id"] = report["id"].id
    return {
        "message": "Report retrieved successfully",
        "data": report,
    }


@app.get("/reports/")
async def get_reports():
    reports = db.select("reports")
    if not reports:
        return {"message": "Reports are not found"}
    for report in reports:
        report["id"] = report["id"].id
    return {
        "message": "Reports retrieved successfully",
        "data": reports,
    }


@app.get("/report/{report_id}/download")
async def download_report(report_id: str) -> bytes:
    report = db.query(
        "SELECT * FROM reports WHERE id = $id",
        {"id": RecordID("reports", report_id)},
    )
    if not report:
        return {"message": "Report not found"}
    report = report[0]
    csv_path = report["csv_path"]

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"{report_id}.csv",
    )
