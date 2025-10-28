import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import scipy

# problem 2:use reduced order model based on POD to solve the 2D Convection-Diffusion Equation

# 步骤：选取快照向量、数值积分格式进而构建快照矩阵S；对S做SVD，取前k个正交基为V；代入计算得q(t)；c(t)=c(0)+V^T q(t)

def POD(h,T,Nsnap,k):
    N = int(1/h)
    dT = T/Nsnap
    csnap = np.loadtxt('csnap.txt')
    S = np.zeros((Nsnap+1,(N+1)*(N+1)))
    for i in range(Nsnap+1):
        if i == 0 or i == Nsnap:
            S[i] = math.sqrt(dT/2) * csnap[i]
        else:
            S[i] = math.sqrt(dT) * csnap[i]
    Ur, Sigma, Vr = scipy.linalg.svd(S.T,full_matrices=True)
    V = Ur[:,:k]
    return V

def F(h,D):
    N = int(1/h)
    mu = D/(h*h)
    f = np.zeros((N+1)*(N+1))
    for j in range(N+1):
        y = (j-0.5)*h
        tep = 0
        if y > 1/3 and y < 2/3:
            tep = 1.0
        else:
            tep = 0.0
        c = (N+1)+j
        f[c] = (mu+1/h)*tep
    return f


def A_x(x, h, D):
    """矩阵乘向量 - 使用虚拟边界方法"""
    N = int(1/h)
    
    # 将向量x转换为矩阵形式 (N+1) x (N+1)
    x_mat = x.reshape(N+1, N+1)
    
    # 创建带虚拟边界的扩展矩阵 (N+2) x (N+2)
    x_ext = np.zeros((N+2, N+2))
    # 物理区域放在 [0:N+1, 0:N+1]，虚拟边界在外围
    x_ext[0:N+1, 0:N+1] = x_mat
    
    # 应用边界条件到虚拟边界
    # 左虚拟边界已经包含在物理区域中，这里处理右虚拟边界和周期性边界
     # 右虚拟边界: 出流条件 (∂c/∂x=0, i=N+1)
    x_ext[N+1, :] = x_ext[N, :]
    
    # 上下虚拟边界: 周期性条件
    # 注意：物理区域是 [0:N+1, 0:N+1]，所以：
    # 下虚拟边界 j=0 对应 j=N
    # 上虚拟边界 j=N+1 对应 j=1
    x_ext[:, N] = x_ext[:, 0]    
    x_ext[:, N+1] = x_ext[:, 1]    # 上虚拟 = 下物理
    
    # 计算A_x
    mu = D/(h*h)
    Ax_ext = np.zeros((N+2, N+2))
    
    # 计算物理区域 (i=0到N, j=0到N)

        
    for i in range(0, N+1):
        for j in range(0, N+1):
            if i != 1:
            # 处理周期性边界的邻居索引
                j_prev = (j-1) % (N+1)  # 下邻居
                j_next = (j+1) % (N+1)  # 上邻居
            
                i_prev = i-1 if i > 0 else -1  # 左邻居
                i_next = i+1 if i < N else N+1  # 右邻居
            
            # 获取邻居值
                left_val = x_ext[i_prev, j] if i > 0 else (1.0 if (1/3 < j*h < 2/3) else 0.0)
                right_val = x_ext[i_next, j]
                up_val = x_ext[i, j_next]
                down_val = x_ext[i, j_prev]
            
            # 扩散项
                diffusion = mu * (left_val + right_val + up_val + down_val - 4*x_ext[i, j])
            # 对流项 (u=1>0, 迎风)
                convection = (1/h) * (left_val - x_ext[i, j])
            else:
                j_prev = (j-1) % (N+1)  # 下邻居
                j_next = (j+1) % (N+1)  # 上邻居
                i_next = i+1
                right_val = x_ext[i_next, j]
                up_val = x_ext[i, j_next]
                down_val = x_ext[i, j_prev]
                diffusion = mu * (right_val + up_val + down_val - 4*x_ext[i, j])
            # 对流项 (u=1>0, 迎风)
                convection = (1/h) * ( - x_ext[i, j])
            
            Ax_ext[i, j] = diffusion + convection
    
    

    # 提取物理区域并转换回向量
    Ax_mat = Ax_ext[0:N+1, 0:N+1]  # 提取物理区域
    Ax_vec = Ax_mat.reshape(-1)     # 展平为一维向量
    
    return Ax_vec

def solve(h,T,Nsnap,k,dt,D):
    N = int (1/h)
    V = POD(h,T,Nsnap,k)
    M = int (T/dt)
    q = np.zeros((M+1,k))
    c = np.zeros((M+1,(N+1)*(N+1)))
    f = F(h,D)
    for j in range(N+1):
        y = (j-0.5)*h
        if y > 1/3 and y < 2/3:
            c[0][j] = 1.0
        else :
            c[0][j] = 0.0
    q[0] = V.T @ c[0]
    for m in range(M):
        qm = q[m]
        Vq = V @ qm
        AVq = A_x(Vq,h,D)
        temp = V.T @ (AVq + f)
        q[m+1] = dt * temp + qm
        c[m+1] = V @ q[m+1]
    C = np.zeros((M+1,N+1,N+1))
    for m in range(M+1):
        C[m] = c[m].reshape(N+1,N+1)
    return C

def visualize(c,dt,k):
    times = [0.2, 0.5, 1.0, 1.5]
    for time_point in times:
        t_index = int(time_point/dt)
        plt.figure()
        plt.imshow(c[t_index].T, extent=[0, 1, 0, 1], origin='lower', cmap='hot', interpolation='nearest', aspect='equal')
        plt.colorbar(label='POD c(t,x,y)')
        plt.title(f'POD k={k} t={time_point} dt={dt}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'POD k={k} t={time_point} dt={dt}.png',dpi=300)
        plt.close()
        print(f'Visualization for k={k} t={time_point} saved as POD t={time_point} dt={dt}.png')
            
if __name__ == "__main__":
    h = 0.005
    T = 1.5
    dt = 0.0005
    D = 0.01
    Nsnap = 100
    k = 1
    c = solve(h,T,Nsnap,k,dt,D)
    visualize(c,dt,k)
    
        
            