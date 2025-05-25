# DNS分类系统

这是一个基于多尺度卷积神经网络的DNS分类系统的Web界面实现。系统支持上传PCAP文件和CSV文件进行分析。

## 功能特点

- 支持拖拽上传文件
- 支持PCAP和CSV文件格式
- 实时显示处理进度
- 可视化展示分类结果
- 响应式界面设计

## 系统要求

- Python 3.7+
- Flask
- TensorFlow
- NumPy
- Pandas
- Scapy

## 安装步骤

1. 克隆项目到本地：
```bash
git clone <repository-url>
cd dns-classification-system
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行系统

1. 启动Flask应用：
```bash
python app.py
```

2. 打开浏览器访问：
```
http://localhost:5000
```

## 使用说明

1. 在网页界面上传文件：
   - 直接拖拽文件到上传区域
   - 或点击上传区域选择文件

2. 支持的文件类型：
   - PCAP文件（.pcap）
   - CSV文件（.csv）

3. 查看结果：
   - 系统会自动处理上传的文件
   - 显示分类结果和置信度
   - 支持批量处理

## 注意事项

- 上传文件大小限制为16MB
- CSV文件格式要求：第一列为标签（0或1），其余列为特征
- 请确保上传文件格式正确

## 目录结构

```
dns-classification-system/
├── app.py              # Flask应用主文件
├── cnn.py             # CNN模型实现
├── requirements.txt    # 项目依赖
├── templates/         # HTML模板
│   └── index.html    # 主页模板
└── uploads/          # 上传文件存储目录
```

## 问题反馈

如有问题或建议，请提交Issue或联系管理员。 