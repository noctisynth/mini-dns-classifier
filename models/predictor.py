import numpy as np

from typing import List, Tuple, Literal
from keras.models import Model
from pydantic import BaseModel
from scapy.all import rdpcap, DNS, PacketList
from datetime import datetime


class PredictionResult(BaseModel):
    timestamp: datetime
    query: str
    prediction: Literal["Covert Channel", "Normal Traffic"]
    confidence: float
    confidence_percent: str


class AnalysisReport(BaseModel):
    total_packets: int
    dns_packets: int
    valid_queries: int
    covert_count: int
    normal_count: int
    top_covert_queries: List[PredictionResult]


class DNSDataProcessor:
    def __init__(self, alphabet: str, max_length: int = 256):
        self.alphabet = alphabet
        self.char_dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self.char_dict["UNK"] = 0
        self.max_length = max_length

    def preprocess(self, text: str) -> np.ndarray:
        indices = np.zeros(self.max_length, dtype=np.int64)
        for i in range(min(len(text), self.max_length)):
            indices[i] = self.char_dict.get(text[i].lower(), 0)
        return indices


class PCAPAnalyzer:
    def __init__(self, model: Model, data_processor: DNSDataProcessor):
        self.model = model
        self.data_processor = data_processor

    def analyze(self, pcap_path: str) -> Tuple[List[PredictionResult], AnalysisReport]:
        packets = rdpcap(pcap_path)
        queries, timestamps = self._extract_dns_queries(packets)
        print(f"Extracted {len(queries)} DNS queries from {len(packets)} packets.")

        processed = [self.data_processor.preprocess(q) for q in queries]
        x = np.array(processed)

        predictions = self.model.predict(x, verbose=0)

        results = self._format_results(queries, timestamps, predictions)
        report = self._generate_report(results, len(packets), len(queries))

        return results, report

    def _extract_dns_queries(self, packets: PacketList) -> Tuple[List[str], List[datetime]]:
        queries, timestamps = [], []
        for pkt in packets:
            if DNS in pkt and pkt[DNS].qr == 0:
                try:
                    query = pkt[DNS].qd.qname.decode("utf-8", "ignore").strip(".")
                    timestamp = datetime.fromtimestamp(float(pkt.time))
                    queries.append(query)
                    timestamps.append(timestamp)
                except Exception:
                    continue
        return queries, timestamps

    def _format_results(
        self, queries, timestamps, predictions
    ) -> List[PredictionResult]:
        return [
            PredictionResult(
                timestamp=ts,
                query=q,
                prediction="Covert Channel" if np.argmax(p) == 1 else "Normal Traffic",
                confidence=float(p.max()),
                confidence_percent=f"{p.max():.2%}",
            )
            for q, ts, p in zip(queries, timestamps, predictions)
        ]

    def _generate_report(
        self, results: List[PredictionResult], total_packets: int, valid_queries: int
    ) -> AnalysisReport:
        covert = [r for r in results if r.prediction == "Covert Channel"]
        top_covert = sorted(covert, key=lambda x: x.confidence, reverse=True)[:5]

        return AnalysisReport(
            total_packets=total_packets,
            dns_packets=len(results),
            valid_queries=valid_queries,
            covert_count=len(covert),
            normal_count=len(results) - len(covert),
            top_covert_queries=top_covert,
        )
