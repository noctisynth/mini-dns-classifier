import numpy as np
import re
import csv
import os
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Embedding
from keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from scapy.all import rdpcap, DNS
import pandas as pd
from datetime import datetime


class CharCNNKim(object):
    def __init__(
        self,
        input_size,
        alphabet_size,
        embedding_size,
        conv_layers,
        fully_connected_layers,
        num_of_classes,
        dropout_p,
        optimizer="adam",
        loss="categorical_crossentropy",
    ):
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self._build_model()

    def _build_model(self):
        # 输入层
        inputs = Input(shape=(self.input_size,), name="sent_input", dtype="int64")

        # 嵌入层（添加L2正则化）
        x = Embedding(
            self.alphabet_size + 1,
            self.embedding_size,
            input_length=self.input_size,
            embeddings_regularizer=l2(1e-4),
        )(inputs)

        # 添加dropout
        x = Dropout(0.2)(x)

        # 多尺度卷积层
        convolution_output = []
        for idx, (num_filters, filter_width) in enumerate(self.conv_layers):
            # 添加L2正则化到卷积层
            conv = Convolution1D(
                filters=num_filters,
                kernel_size=filter_width,
                activation="relu",
                padding="same",
                kernel_regularizer=l2(1e-4),
                name=f"Conv1D_{num_filters}_{filter_width}_{idx}",  # 添加索引确保名称唯一
            )(x)

            # 添加批归一化
            conv = BatchNormalization(
                name=f"BatchNorm_{num_filters}_{filter_width}_{idx}"
            )(conv)

            # 使用最大池化和平均池化的组合
            max_pool = GlobalMaxPooling1D(
                name=f"MaxPoolingOverTime_{num_filters}_{filter_width}_{idx}"
            )(conv)
            avg_pool = GlobalAveragePooling1D(
                name=f"AvgPoolingOverTime_{num_filters}_{filter_width}_{idx}"
            )(conv)

            convolution_output.extend([max_pool, avg_pool])

        # 合并所有池化输出
        x = Concatenate(name="concatenate_pools")(convolution_output)

        # 全连接层
        for i, fl in enumerate(self.fully_connected_layers):
            x = Dense(
                fl,
                activation="relu",
                kernel_regularizer=l2(1e-4),
                name=f"dense_{fl}_{i}",
            )(x)  # 添加索引确保名称唯一
            x = BatchNormalization(name=f"BatchNorm_dense_{fl}_{i}")(x)
            x = Dropout(self.dropout_p, name=f"dropout_{fl}_{i}")(x)

        # 输出层
        predictions = Dense(
            self.num_of_classes,
            activation="softmax",
            kernel_regularizer=l2(1e-4),
            name="output",
        )(x)

        # 构建模型
        model = Model(inputs=inputs, outputs=predictions)

        # 使用Adam优化器，并设置学习率
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,  # 梯度裁剪
        )

        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )

        self.model = model
        print("CharCNNKim model built: ")
        self.model.summary()

    def train(
        self,
        training_inputs,
        training_labels,
        validation_inputs,
        validation_labels,
        epochs,
        batch_size,
        checkpoint_every=100,
    ):
        print("Training CharCNNKim model: ")

        # 创建检查点目录
        checkpoint_dir = "./checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 创建日志目录
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 回调函数
        callbacks = [
            # 早停
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
            ),
            # 模型检查点
            ModelCheckpoint(
                os.path.join(checkpoint_dir, "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            # 学习率调度器
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            # TensorBoard
            TensorBoard(
                log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch"
            ),
        ]

        # 训练模型
        history = self.model.fit(
            training_inputs,
            training_labels,
            validation_data=(validation_inputs, validation_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        return history


class Data(object):
    def __init__(self, data_source, alphabet, input_size=256, num_of_classes=2):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.dict["UNK"] = 0
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        self.data_source = data_source
        self.num_of_classes = num_of_classes

    def load_data(self):
        data = []
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                rdr = csv.reader(f, delimiter=",", quotechar='"')
                for row in rdr:
                    if len(row) < 2:  # 跳过格式不正确的行
                        continue
                    txt = ""
                    for s in row[1:]:
                        txt = (
                            txt
                            + " "
                            + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                        )
                    try:
                        label = int(row[0])
                        if label not in [0, 1]:  # 确保标签是0或1
                            continue
                        data.append((label, txt.strip()))
                    except ValueError:
                        continue  # 跳过标签无效的行
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return

        self.data = np.array(data)
        print(f"Data loaded from {self.data_source} with {len(self.data)} samples")

        # 打印类别分布
        labels = [x[0] for x in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples ({c / len(labels) * 100:.2f}%)")

    def get_all_data(self):
        if not hasattr(self, "data"):
            print("No data loaded. Please call load_data() first.")
            return None, None

        batch_indices = []
        one_hot = np.eye(self.num_of_classes, dtype="int64")
        classes = []

        for c, s in self.data:
            batch_indices.append(self.str_to_indexes(s))
            c = int(c)
            classes.append(one_hot[c])

        return np.asarray(batch_indices, dtype="int64"), np.asarray(classes)

    def str_to_indexes(self, s):
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype="int64")
        for i in range(max_length):
            c = s[i]
            str2idx[i] = self.dict.get(c, 0)  # 使用get方法，对于未知字符返回0
        return str2idx


def process_pcap_file(pcap_path, model, alphabet):
    """处理PCAP文件并进行预测"""
    try:
        # 读取PCAP文件
        print(f"\n[+] 正在读取PCAP文件: {pcap_path}")
        packets = rdpcap(pcap_path)

        # 提取DNS查询
        dns_queries = []
        timestamps = []

        print("[*] 开始解析数据包...")
        total_packets = len(packets)
        dns_count = 0
        error_count = 0

        for i, packet in enumerate(packets):
            try:
                # 检查是否包含DNS层
                if DNS in packet:
                    dns_count += 1
                    # 检查是否为DNS查询（不是响应）
                    if hasattr(packet[DNS], "qr") and packet[DNS].qr == 0:
                        # 确保包含查询部分
                        if hasattr(packet[DNS], "qd") and packet[DNS].qd is not None:
                            for qry in packet[DNS].qd:
                                try:
                                    # 尝试获取查询名称
                                    if hasattr(qry, "qname"):
                                        query = qry.qname
                                        if isinstance(query, bytes):
                                            query = query.decode(
                                                "utf-8", errors="ignore"
                                            )
                                        elif isinstance(query, str):
                                            pass
                                        else:
                                            query = str(query)

                                        # 处理时间戳
                                        try:
                                            if hasattr(packet, "time"):
                                                # 确保时间戳是数值类型
                                                packet_time = float(packet.time)
                                                timestamp = datetime.fromtimestamp(
                                                    packet_time
                                                )
                                            else:
                                                timestamp = datetime.now()

                                            dns_queries.append(query)
                                            timestamps.append(timestamp)
                                        except (ValueError, TypeError) as e:
                                            print(f"[!] 时间戳解析错误: {str(e)}")
                                            timestamp = datetime.now()
                                            dns_queries.append(query)
                                            timestamps.append(timestamp)
                                except Exception:
                                    error_count += 1
                                    continue
            except Exception:
                error_count += 1
                continue

            # 每处理1000个包打印一次进度
            if (i + 1) % 1000 == 0:
                print(f"[*] 已处理 {i + 1}/{total_packets} 个数据包...")

        print("\n[*] 数据包解析完成:")
        print(f"    - 总数据包数: {total_packets}")
        print(f"    - DNS数据包数: {dns_count}")
        print(f"    - 解析错误数: {error_count}")

        if not dns_queries:
            print("[!] 未在PCAP文件中找到有效的DNS查询")
            return

        print(f"[+] 发现 {len(dns_queries)} 个有效的DNS查询")

        # 准备数据进行预测
        data_processor = Data("", alphabet)  # 创建数据处理器实例
        processed_queries = []
        valid_queries = []
        valid_timestamps = []

        print("[*] 处理DNS查询数据...")
        for query, timestamp in zip(dns_queries, timestamps):
            try:
                # 清理和验证查询字符串
                query = query.strip().strip(".")
                if query and len(query) > 0:
                    processed_query = data_processor.str_to_indexes(query)
                    processed_queries.append(processed_query)
                    valid_queries.append(query)
                    valid_timestamps.append(timestamp)
            except Exception:
                continue

        if not processed_queries:
            print("[!] 没有可用于预测的有效DNS查询")
            return

        processed_queries = np.array(processed_queries)

        # 进行预测
        print("[+] 正在进行预测...")
        predictions = model.predict(processed_queries, verbose=0)

        # 准备结果数据
        results = []
        for query, pred, timestamp in zip(valid_queries, predictions, valid_timestamps):
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]
            results.append(
                {
                    "timestamp": timestamp,
                    "query": query,
                    "prediction": "Covert Channel"
                    if pred_class == 1
                    else "Normal Traffic",
                    "confidence": f"{confidence:.2%}",
                }
            )

        # 将结果保存为CSV文件
        output_dir = "predictions"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"prediction_results_{timestamp_str}.csv"
        )

        # 使用utf-8-sig编码（带BOM），确保Excel正确识别编码
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        # 创建一个更易读的报告文件
        report_file = os.path.join(output_dir, f"analysis_report_{timestamp_str}.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("DNS流量分析报告\n")
            f.write("=" * 50 + "\n\n")

            # 写入统计信息
            covert_count = sum(
                1 for r in results if r["prediction"] == "Covert Channel"
            )
            normal_count = len(results) - covert_count

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PCAP文件: {os.path.basename(pcap_path)}\n\n")
            f.write("解析统计:\n")
            f.write(f"- 总数据包数: {total_packets}\n")
            f.write(f"- DNS数据包数: {dns_count}\n")
            f.write(f"- 解析错误数: {error_count}\n")
            f.write(f"- 有效DNS查询数: {len(valid_queries)}\n\n")
            f.write("预测统计:\n")
            f.write(f"- 检测到的隐蔽信道查询数: {covert_count}\n")
            f.write(f"- 正常查询数: {normal_count}\n\n")

            # 写入可疑查询详情
            if covert_count > 0:
                f.write("可疑的DNS查询 (置信度最高的前5个):\n")
                suspicious = sorted(
                    [r for r in results if r["prediction"] == "Covert Channel"],
                    key=lambda x: float(x["confidence"].strip("%")),
                    reverse=True,
                )
                for i, r in enumerate(suspicious[:5]):
                    f.write(f"\n{i + 1}. 查询: {r['query']}\n")
                    f.write(f"   时间: {r['timestamp']}\n")
                    f.write(f"   置信度: {r['confidence']}\n")

        print("\n[+] 分析完成！")
        print(f"[+] 详细结果已保存至: {output_file}")
        print(f"[+] 分析报告已保存至: {report_file}")

        # 打印统计信息
        print("\n统计信息:")
        print(f"总数据包数: {total_packets}")
        print(f"DNS数据包数: {dns_count}")
        print(f"有效DNS查询数: {len(valid_queries)}")
        print(f"检测到的隐蔽信道查询数: {covert_count}")
        print(f"正常查询数: {normal_count}")

        # 打印可疑样本
        if covert_count > 0:
            print("\n可疑的DNS查询 (置信度最高的前5个):")
            for i, r in enumerate(suspicious[:5]):
                print(f"{i + 1}. 查询: {r['query']}")
                print(f"   时间: {r['timestamp']}")
                print(f"   置信度: {r['confidence']}")

    except Exception as e:
        print(f"[!] 处理PCAP文件时出错: {str(e)}")
        import traceback

        print(traceback.format_exc())


def select_pcap_file():
    """打开文件选择对话框选择PCAP文件"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择PCAP文件",
        filetypes=[("PCAP files", "*.pcap *.pcapng"), ("All files", "*.*")],
    )
    return file_path if file_path else None


def main():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)

    # 初始化参数
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    try:
        # 初始化模型
        model = CharCNNKim(
            input_size=256,
            alphabet_size=len(alphabet),
            embedding_size=128,
            conv_layers=[[256, 7], [256, 5], [256, 3]],
            fully_connected_layers=[1024, 512],
            num_of_classes=2,
            dropout_p=0.5,
            optimizer="adam",
            loss="categorical_crossentropy",
        )

        # 检查是否存在已训练的模型权重
        weights_path = "model_weights.h5"
        if os.path.exists(weights_path):
            print("[+] 加载已训练的模型权重...")
            try:
                model.model.load_weights(weights_path)
            except Exception as e:
                print(f"[!] 加载模型权重失败: {str(e)}")
                print("[*] 将开始训练新模型...")
                # 加载训练数据
                training_data = Data("train.csv", alphabet, 256, 2)
                training_data.load_data()
                training_inputs, training_labels = training_data.get_all_data()

                # 从训练数据中分割出一部分作为验证集
                train_inputs, val_inputs, train_labels, val_labels = train_test_split(
                    training_inputs,
                    training_labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=np.argmax(training_labels, axis=1),
                )

                # 训练模型
                model.train(
                    training_inputs=train_inputs,
                    training_labels=train_labels,
                    validation_inputs=val_inputs,
                    validation_labels=val_labels,
                    epochs=8,
                    batch_size=32,
                )

                # 保存模型权重
                model.model.save_weights(weights_path)
                print(f"\n[+] 模型训练完成并保存权重至: {weights_path}")
        else:
            print("[+] 未找到已训练的模型权重，开始训练新模型...")
            # 加载训练数据
            training_data = Data("train.csv", alphabet, 256, 2)
            training_data.load_data()
            training_inputs, training_labels = training_data.get_all_data()

            # 从训练数据中分割出一部分作为验证集
            train_inputs, val_inputs, train_labels, val_labels = train_test_split(
                training_inputs,
                training_labels,
                test_size=0.2,
                random_state=42,
                stratify=np.argmax(training_labels, axis=1),
            )

            # 训练模型
            model.train(
                training_inputs=train_inputs,
                training_labels=train_labels,
                validation_inputs=val_inputs,
                validation_labels=val_labels,
                epochs=8,
                batch_size=32,
            )

            # 保存模型权重
            model.model.save_weights(weights_path)
            print(f"\n[+] 模型训练完成并保存权重至: {weights_path}")

        # 选择PCAP文件
        print("\n[*] 请选择要分析的PCAP文件...")
        pcap_path = select_pcap_file()

        if pcap_path:
            # 处理PCAP文件
            process_pcap_file(pcap_path, model.model, alphabet)
        else:
            print("[!] 未选择PCAP文件，程序退出")

    except Exception as e:
        print(f"[!] 程序执行出错: {str(e)}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
