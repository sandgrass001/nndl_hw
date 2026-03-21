import numpy as np
import matplotlib.pyplot as plt
# ============================
# 1. 目标函数：定义待拟合的复杂非线性函数
# 融合三角函数和二次多项式，提升拟合难度，验证ReLU网络万能逼近性
# ============================
def target_function(x):
    return np.sin(3*x) + 0.5*np.cos(5*x) + 0.3*x**2 + 0.2*np.sin(10*x)
# ============================
# 2. Fourier特征增强：将一维输入映射到高维特征空间
# 目的是降低非线性拟合难度，让模型更易捕捉函数特征
# ============================
def feature_map(x):
    return np.hstack([
        x,
        np.sin(x), np.cos(x),
        np.sin(3*x), np.cos(3*x),
        np.sin(5*x), np.cos(5*x),
        np.sin(10*x), np.cos(10*x)
    ])
# ============================
# 3. 数据生成与预处理：划分训练集/测试集，归一化（关键步骤）
# 设置随机种子，保证实验可复现
# ============================
np.random.seed(42)
# 训练集：[-2,2]区间均匀采样200个点，reshape为(-1,1)适配网络输入格式
x_train = np.linspace(-2, 2, 200).reshape(-1, 1)
y_train = target_function(x_train)
# 测试集：同区间采样400个点（更高密度），用于验证泛化能力
x_test = np.linspace(-2, 2, 400).reshape(-1, 1)
y_test = target_function(x_test)
# 对输入进行特征增强
X_train = feature_map(x_train)
X_test = feature_map(x_test)
# 🔥 输入归一化（关键）：消除特征尺度差异，加速梯度下降收敛
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8  # 加1e-8避免分母为0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
# ============================
# 4. ReLU激活函数及其梯度：网络核心激活单元
# ReLU解决梯度消失问题，适合深层（此处为两层）网络
# ============================
def relu(x):
    return np.maximum(0, x)  # ReLU公式：max(0,x)
def relu_grad(x):
    return (x > 0).astype(float)  # ReLU梯度：x>0时为1，否则为0
# ============================
# 5. 两层ReLU网络类：纯NumPy实现，无框架依赖（附加分要求）
# 包含前向传播（计算预测值）和反向传播（更新参数）
# ============================
class TwoLayerReLU:
    def __init__(self, in_dim, hidden_dim, out_dim):
        # He初始化：适配ReLU，避免梯度消失，参数服从正态分布
        self.W1 = np.random.randn(in_dim, hidden_dim) * np.sqrt(2/in_dim)
        self.b1 = np.zeros((1, hidden_dim))  # 偏置初始化为0
        self.W2 = np.random.randn(hidden_dim, out_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros((1, out_dim))
    def forward(self, x):
        # 前向传播：输入→隐藏层（ReLU激活）→输出层
        self.z1 = x @ self.W1 + self.b1  # 隐藏层线性输出
        self.a1 = relu(self.z1)  # 隐藏层ReLU激活输出
        self.z2 = self.a1 @ self.W2 + self.b2  # 输出层线性输出（回归无激活）
        return self.z2
    def backward(self, x, y, y_pred, lr):
 # 反向传播：计算梯度并更新参数，使用批量梯度下降（BGD）
        m = x.shape[0]  # 样本数量
        # 输出层梯度计算
        dz2 = (y_pred - y) / m  # 损失对z2的梯度（MSE损失导数）
        dW2 = self.a1.T @ dz2  # W2的梯度
        db2 = np.sum(dz2, axis=0, keepdims=True)  # b2的梯度
        # 隐藏层梯度计算（链式法则）
        da1 = dz2 @ self.W2.T  # 损失对a1的梯度
        dz1 = da1 * relu_grad(self.z1)  # 损失对z1的梯度（乘ReLU梯度）
        dW1 = x.T @ dz1  # W1的梯度
        db1 = np.sum(dz1, axis=0, keepdims=True)  # b1的梯度
        # 参数更新：梯度下降，学习率为lr
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
# ============================
# 6. 模型训练：初始化模型，迭代训练并记录损失
# ============================
# 初始化模型：输入维度9（特征增强后），隐藏层256个神经元，输出维度1
model = TwoLayerReLU(X_train.shape[1], 256, 1)
epochs = 6000  # 训练轮数
lr = 0.01  # 初始学习率
loss_history = []  # 记录每轮损失，用于后续绘图
for epoch in range(epochs):
 # 🔥 学习率衰减：每500轮衰减一次，避免后期震荡，提升收敛稳定性
    lr_decay = lr * (0.98 ** (epoch // 500))
    # 前向传播：计算训练集预测值
    y_pred = model.forward(X_train)
    # 计算均方误差（MSE）损失
    loss = np.mean((y_pred - y_train)**2)
    loss_history.append(loss)
    # 反向传播：更新模型参数
    model.backward(X_train, y_train, y_pred, lr_decay)
    # 每500轮打印一次损失，观察训练进度
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
# ============================
# 7. 模型测试与可视化：验证拟合效果，绘制拟合曲线和损失曲线
# ============================
y_test_pred = model.forward(X_test)
# 设置画布大小，绘制两个子图
plt.figure(figsize=(12,5))
# 子图1：测试集真实值与预测值对比（拟合效果）
plt.subplot(1,2,1)
plt.plot(x_test, y_test, label="True")  # 真实函数曲线
plt.plot(x_test, y_test_pred, '--', label="Pred")  # 模型预测曲线
plt.legend()  # 显示图例
plt.title("Two-Layer ReLU (Improved)")  # 子图标题
# 子图2：损失曲线（训练过程）
plt.subplot(1,2,2)
plt.plot(loss_history)  # 绘制损失变化
plt.title("Loss Curve")  # 子图标题
plt.show()  # 显示图像