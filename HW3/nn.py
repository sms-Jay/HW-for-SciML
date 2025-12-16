import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示

np.random.seed(42)
torch.manual_seed(42)

class TwoLayerNet(nn.Module):
    def __init__(self, D):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(1, D)  # 第一层
        self.fc2 = nn.Linear(D, 1)  # 第二层
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepNet(nn.Module):
    def __init__(self, D, hidden_layers=2):
        super(DeepNet, self).__init__()
        layers = []
        
        # 输入层
        layers.append(nn.Linear(1, D))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(D, D))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(D, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def plot_results(x_test, y_pred, y_true, losses, D, func_type, is_deep=False, layers=0):
    
    # 计算误差
    error = y_pred - y_true
    
    if is_deep:
        # 深度网络：绘制4个子图
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        
        # 整个区间
        axes[0].plot(x_test, y_pred, 'b-', linewidth=1.5, alpha=0.8, label='Prediction')
        axes[0].plot(x_test, y_true, 'r--', linewidth=1, alpha=0.6, label='Truth')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'Full Range\nD={D}, Layers={layers}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 训练区间放大
        train_mask = (x_test >= -1) & (x_test <= 1)
        axes[1].plot(x_test[train_mask], y_pred[train_mask], 'b-', linewidth=2, label='Prediction')
        axes[1].plot(x_test[train_mask], y_true[train_mask], 'r--', linewidth=1.5, alpha=0.7, label='Truth')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Training Region [-1, 1]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 误差
        axes[2].plot(x_test, error, 'g-', linewidth=1)
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('Error')
        axes[2].set_title('Prediction Error')
        axes[2].grid(True, alpha=0.3)
        
        # 损失曲线
        axes[3].plot(losses, 'm-', linewidth=1)
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss')
        axes[3].set_title('Training Loss')
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)
        
    else:
        # 浅层网络：绘制3个子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 预测结果
        axes[0].plot(x_test, y_pred, 'b-', label='Prediction', linewidth=2)
        axes[0].plot(x_test, y_true, 'r--', label='Truth', linewidth=2, alpha=0.7)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'Prediction vs Truth\nD={D}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 误差
        axes[1].plot(x_test, error, 'g-', linewidth=2)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Prediction Error')
        axes[1].grid(True, alpha=0.3)
        
        # 损失曲线
        axes[2].plot(losses, 'm-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()

    if is_deep:
        filename = f'sine D{D} L{layers}.png'
    else:
        filename = f'{func_type} D{D}.png'
    
    # 保存图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # 显示图片
    # plt.show()
    
    # 关闭图形释放内存
    plt.close()
    
    return error, train_mask if is_deep else (x_test >= -1) & (x_test <= 1)

def train_and_test(D, func_type='linear', is_deep=False, layers=0):

    # 生成训练数据
    n_samples = 1000
    x_train = torch.rand(n_samples) * 2 - 1  # [-1, 1]
    
    if func_type == 'linear':
        y_train = x_train
    elif func_type == 'quadratic':
        y_train = x_train**2
    elif func_type == 'sine':
        y_train = torch.sin(10 * np.pi * x_train)
    
    # 创建模型
    if is_deep:
        model = DeepNet(D, layers)
        lr = 0.001
        epochs = 10000
    else:
        model = TwoLayerNet(D)
        if func_type == 'linear':
            lr = 0.001
            epochs = 2000
        elif func_type == 'quadratic':
            lr = 0.0001
            epochs = 5000
        elif func_type == 'sine':
            lr = 0.00001
            epochs = 10000
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train.view(-1, 1))
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # 测试
    x_test = torch.linspace(-10, 10, 1000).view(-1, 1)
    y_pred = model(x_test)
    
    # 计算真实值
    if func_type == 'linear':
        y_true = x_test
    elif func_type == 'quadratic':
        y_true = x_test**2
    elif func_type == 'sine':
        y_true = torch.sin(10 * np.pi * x_test)
    
    return x_test.numpy(), y_pred.detach().numpy(), y_true.numpy(), losses

def analyze_error(error, x_test):

    # 创建掩码
    train_mask = (x_test >= -1) & (x_test <= 1)
    test_mask = ~train_mask
    
    # 分割误差
    train_error = error[train_mask]
    test_error = error[test_mask]
    
    # 计算统计量
    train_max_error = np.max(np.abs(train_error)) if len(train_error) > 0 else 0
    train_mean_error = np.mean(np.abs(train_error)) if len(train_error) > 0 else 0
    test_max_error = np.max(np.abs(test_error)) if len(test_error) > 0 else 0
    test_mean_error = np.mean(np.abs(test_error)) if len(test_error) > 0 else 0
    
    # 打印结果
    print("Inside training region [-1, 1]:")
    print(f"  Max error: {train_max_error:.6f}")
    print(f"  Mean error: {train_mean_error:.6f}")
    
    print("Outside training region:")
    print(f"  Max error: {test_max_error:.6f}")
    print(f"  Mean error: {test_mean_error:.6f}")
    
    # 计算过拟合比（避免除以0）
    if train_mean_error > 1e-10:
        overfit_ratio = test_mean_error / train_mean_error
        print(f"Overfitting ratio: {overfit_ratio:.2f}")
    else:
        print("Overfitting ratio: N/A (training error too small)")
    
    return 

def main():
    
    # 1.1线性函数
    print("Experiment 1: Linear Function y = x")
    func_type = 'linear'
    for D in [2, 5, 10, 20, 100, 500]:
        print(f"\nTraining with D={D}...")
        x_test, y_pred, y_true, losses = train_and_test(D, 'linear')
        error, train_mask = plot_results(x_test, y_pred, y_true, losses, D, func_type)
        analyze_error(error, x_test.flatten())
    
    # 1.2 二次函数
    print("Experiment 2: Quadratic Function y = x^2")
    func_type = 'quadratic'
    for D in [2, 10, 50, 100, 500, 1000]:
        print(f"\nTraining with D={D}...")
        x_test, y_pred, y_true, losses = train_and_test(D, 'quadratic')
        error, train_mask = plot_results(x_test, y_pred, y_true, losses, D, func_type)
        analyze_error(error, x_test.flatten())
    
    # 1.3 高频正弦函数
    print("Experiment 3: High-frequency Sine Function y = sin(10πx)")
    func_type = 'sine'
    
    # 使用浅层网络
    for D in [2, 10, 50, 100, 500, 1000]:
        print(f"\nTraining network with D={D}...")
        x_test, y_pred, y_true, losses = train_and_test(D, 'sine')
        error, train_mask = plot_results(x_test, y_pred, y_true, losses, D, func_type)
        analyze_error(error, x_test.flatten())
    
    # 使用深度网络
    """
    configs = [(50, 3), (100, 4), (200, 5), (500, 10)]
    for D, layers in configs:
        print(f"\nTraining deep network with D={D}, Layers={layers}...")
        x_test, y_pred, y_true, losses = train_and_test(D, 'sine', is_deep=True, layers=layers)
        error, train_mask = plot_results(x_test, y_pred, y_true, losses, D, func_type, is_deep=True, layers=layers)
        analyze_error(error, x_test.flatten())
    """
if __name__ == "__main__":
    main()
    