import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import filedialog
from scapy.all import rdpcap, DNS
from scapy.config import conf
import numpy as np
from keras.models import load_model

conf.use_pcap = False

# ==== 定义字符表（与训练完全一致） ====
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
CHAR_DICT = {'UNK': 0}
for idx, char in enumerate(ALPHABET):
    CHAR_DICT[char] = idx + 1


def str_to_indexes(s, input_size=256):  # 与训练时一致
    """从头部开始截断域名"""
    s = s.lower()
    max_length = min(len(s), input_size)
    str2idx = np.zeros(input_size, dtype='int64')
    for i in range(max_length):
        c = s[i]
        str2idx[i] = CHAR_DICT.get(c, 0)
    return str2idx


def extract_char_features(pkt):
    dns = pkt.getlayer(DNS)
    if not dns or not dns.qd:
        return None
    try:
        qname = dns.qd.qname.decode('utf-8', errors='replace')
        return str_to_indexes(qname)
    except Exception as e:
        return None


def load_model_and_detect():
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="选择模型文件", filetypes=[("HDF5模型文件", "*.h5")])
    if not model_path:
        return

    pcap_path = filedialog.askopenfilename(title="选择PCAP文件", filetypes=[("PCAP文件", "*.pcap")])
    if not pcap_path:
        return

    try:
        model = load_model(model_path)
        print(f"模型加载成功: {model_path}")
        print(f"模型输入形状: {model.input_shape}")  # 验证输入形状
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    packets = rdpcap(pcap_path)
    threshold = 0.5  # 初始阈值设为0.5
    suspicious = []

    for pkt in packets:
        if pkt.haslayer(DNS) and pkt.dport == 53:
            indexes = extract_char_features(pkt)
            if indexes is not None:
                input_data = np.array([indexes], dtype='int64')
                print(f"输入数据形状: {input_data.shape}")  # 应为 (1, 256)
                prob = model.predict(input_data, verbose=0)[0][1]
                if prob > threshold:
                    suspicious.append((pkt.time, indexes, prob))

    print(f"\n=== 检测结果 ===")
    print(f"阈值: {threshold}, 可疑流量数: {len(suspicious)}")
    for idx, (ts, idxs, prob) in enumerate(suspicious, 1):
        domain = ''.join([ALPHABET[i - 1] if i != 0 else '?' for i in idxs if i != 0])
        print(f"\n[可疑流量 {idx}]")
        print(f"时间戳: {ts}")
        print(f"异常概率: {prob:.4f}")
        print(f"域名: {domain}")


if __name__ == "__main__":
    load_model_and_detect()