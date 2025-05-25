import numpy as np
import tensorflow as tf

from typing import List, Tuple
from .model import CharCNNKim
from .predictor import PCAPAnalyzer, DNSDataProcessor, PredictionResult, AnalysisReport

tf.get_logger().setLevel("ERROR")


def initialize_model(
    weights_path: str = "model_weights.h5",
) -> Tuple[CharCNNKim, DNSDataProcessor]:
    np.random.seed(42)
    tf.random.set_seed(42)

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    model = CharCNNKim()
    model.load_weights(weights_path)

    processor = DNSDataProcessor(alphabet)
    return model, processor


def analyze_pcap(
    model: CharCNNKim,
    processor: DNSDataProcessor,
    pcap_path: str,
) -> Tuple[List[PredictionResult], AnalysisReport]:
    analyzer = PCAPAnalyzer(model.model, processor)
    return analyzer.analyze(pcap_path)
