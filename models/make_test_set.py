import pandas as pd

NAMES_PATHS = [
    "data/dns2tcp/names.csv",
    "data/dnscapy/names.csv",
    "data/iodine/names.csv",
    "data/plain/names.csv",
    "data/tuns/names.csv",
    "data/DomainFrontingLists/Google-hosted.txt",
    "data/DomainFrontingLists/Cloudfront.txt",
]

TRAIN_PATHS = [
    "train.csv",
    "validate.csv",
]

# 合并所有候选域名数据
dataframes = []
for path in NAMES_PATHS:
    df = pd.read_csv(path)
    dataframes.append(df)
names_df = pd.concat(dataframes, ignore_index=True)

# 合并训练数据
train_dataframes = []
for path in TRAIN_PATHS:
    df = pd.read_csv(f"{path}", header=None, names=["label", "qname"])
    df = df.set_index("qname")
    train_dataframes.append(df)
train_df = pd.concat(train_dataframes)

# 生成测试集
test_df = names_df[~names_df["qname"].isin(train_df.index)]
test_df = test_df.sample(20000)
test_df.to_csv("multilabel/test.csv", index=False, header=False)