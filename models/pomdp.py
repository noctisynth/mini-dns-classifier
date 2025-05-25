import enum
import numpy as np
import time
import argparse
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 设置matplotlib支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 枚举定义
class ActionType(enum.Enum):
    ANALYZE = 0
    BLOCK = 1
    PASS = 2


class ObservationType(enum.Enum):
    OK = 0
    BAD = 1


class StateType(enum.Enum):
    REGULAR = 0
    TUNNEL = 1


class DnsTunnelModel:
    """DNS隧道环境模型"""
    def __init__(self, Q, L):
        self.Q = Q
        self.L = L
        self._states = [s.value for s in StateType]
        self._actions = [a.value for a in ActionType]
        self._observations = [o.value for o in ObservationType]
        self.state = np.random.choice(self._states)  # 初始状态

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def observations(self):
        return self._observations

    def transition(self, state, action):
        if action == ActionType.ANALYZE.value:
            return state
        else:
            return np.random.choice(self._states)

    def observation(self, state, action):
        if action == ActionType.ANALYZE.value:
            if state == StateType.REGULAR.value:
                return ObservationType.OK.value if np.random.rand() < self.Q else ObservationType.BAD.value
            else:
                return ObservationType.BAD.value if np.random.rand() < self.Q else ObservationType.OK.value
        else:
            return np.random.choice(self._observations)

    def reward(self, state, action, next_state):
        if action == ActionType.ANALYZE.value:
            return self.L * 0.8
        elif action == ActionType.BLOCK.value:
            return self.L if state == StateType.TUNNEL.value else -(1 - self.L)
        elif action == ActionType.PASS.value:
            return self.L if state == StateType.REGULAR.value else -(1 - self.L)

    def step(self, action):
        """执行一步操作，返回下一个状态、观察和奖励"""
        next_state = self.transition(self.state, action)
        obs = self.observation(next_state, action)
        reward = self.reward(self.state, action, next_state)
        self.state = next_state
        return next_state, obs, reward


class Environment:
    """环境状态封装"""
    def __init__(self, model):
        self.model = model
        self.state = np.random.choice(model.states)

    def step(self, action):
        next_state = self.model.transition(self.state, action)
        obs = self.model.observation(next_state, action)
        reward = self.model.reward(self.state, action, next_state)
        self.state = next_state
        return next_state, obs, reward


def evaluate_model(Q, L, num_steps=500):
    """单组参数模型评估"""
    model = DnsTunnelModel(Q, L)

    # 收集状态-动作-观察对数据
    data = []
    labels = []

    for _ in range(num_steps):
        action = np.random.choice(model.actions)
        next_state, obs, reward = model.step(action)

        data.append([action, obs])  # 特征：动作和观察
        labels.append(1 if next_state == StateType.TUNNEL.value else 0)  # 标签：1为隧道，0为常规

    # 转换数据为numpy数组
    X = np.array(data)
    y = np.array(labels)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 切分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测并计算性能
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    accuracy = np.mean(y_pred == y_test)

    return fpr, tpr, roc_auc, accuracy


def plot_results(Q=0.85, L_start=0.2, L_end=1.0, num_points=9):
    """批量多L值评估 + 绘图"""
    L_values = np.linspace(L_start, L_end, num_points)

    # 初始化存储结果
    auc_values = []
    accuracies = []

    for L in L_values:
        fpr, tpr, roc_auc, accuracy = evaluate_model(Q, L)
        auc_values.append(roc_auc)
        accuracies.append(accuracy)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'L={L:.1f} (AUC={roc_auc:.2f})')

    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('不同安全等级下的ROC曲线', fontsize=14)
    plt.legend()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(L_values, accuracies, 'bo-', label='逻辑回归模型')
    plt.axhline(Q, color='r', linestyle='--', label='基础检测器')
    plt.xlabel('安全等级 (L)', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('准确率与安全等级的关系', fontsize=14)
    plt.legend()
    plt.savefig('accuracy_curve.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNS隧道检测模拟')
    parser.add_argument("--Q", type=float, default=0.85, help="检测器基础准确率 (默认0.85)")
    parser.add_argument("--L_start", type=float, default=0.2, help="安全等级起始值 (默认0.2)")
    parser.add_argument("--L_end", type=float, default=1.0, help="安全等级结束值 (默认1.0)")
    parser.add_argument("--steps", type=int, default=500, help="模拟步数 (默认500)")
    args = parser.parse_args()

    print(f"""
========== 参数配置 ==========
检测器准确率(Q): {args.Q}
安全等级范围(L): {args.L_start} ~ {args.L_end}
模拟步数: {args.steps}
=============================
""")

    print("启动模拟...")
    start_time = time.time()

    plot_results(Q=args.Q, L_start=args.L_start, L_end=args.L_end)

    print(f"模拟完成，耗时 {time.time() - start_time:.2f} 秒")
    print("结果已保存至: roc_curve.png 和 accuracy_curve.png")
