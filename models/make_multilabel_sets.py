from functools import reduce
import pandas as pd

PATHS = [
    "data/dns2tcp/2018-03-23-11-08-11.csv",
    "data/dnscapy/2018-03-29-19-06-25.csv",
    "data/iodine/2018-03-19-19-06-24.csv",
    "data/plain/2018-03-19-19-34-33.csv",
    "data/tuns/2018-03-30-09-40-10.csv",
]

# 读取所有CSV文件到DataFrame列表
dataframes = [pd.read_csv(path) for path in PATHS]

# 计算最小样本量并裁剪
mincount = min(df.shape[0] for df in dataframes)
dataframes = [df.sample(mincount) for df in dataframes]

# 合并所有DataFrame（修复点）
names_df = pd.concat(dataframes, ignore_index=True)

# 生成训练集、验证集、测试集
train_df = names_df.sample(16000, random_state=1)
validate_df = names_df.sample(5000, random_state=2)
test_df = names_df.sample(20000, random_state=3)

# 保存结果
train_df.to_csv("multilabel/train.csv", index=False, header=True)
validate_df.to_csv("multilabel/validate.csv", index=False, header=True)
test_df.to_csv("multilabel/test.csv", index=False, header=True)
