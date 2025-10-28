import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

# problem 1:use finite volume method to solve the 2D Convection-Diffusion Equation
# Default: dx = dy = h

def FVM(h, T, dt, D):
    N = int(1/h) 
    M = int(T/dt)
    mu_diff = D * dt / (h*h)
    mu_conv = dt / h
    
    c = np.zeros((M+1, N+2, N+2))  # 增加虚拟边界
    
    for t in range(1, M+1):
        # 先复制，然后计算内部点
        c[t] = c[t-1].copy()
        
        # 内部点计算 (i=1到N, j=1到N)
        for i in range(1, N+1):
            for j in range(1, N+1):
                # 扩散项
                diff = (c[t-1,i-1,j] + c[t-1,i+1,j] + 
                       c[t-1,i,j-1] + c[t-1,i,j+1] - 4*c[t-1,i,j])
                # 对流项
                conv = c[t-1,i-1,j] - c[t-1,i,j]  # u=1>0, 迎风
                
                c[t,i,j] = c[t-1,i,j] + mu_diff * diff + mu_conv * conv
        
        # 边界条件（计算完成后施加）
        # 左边界：入流
        for j in range(1, N+1):
            y = (j-0.5)*h  # 单元中心坐标
            if 1/3 < y < 2/3:
                c[t,0,j] = 1.0
            else:
                c[t,0,j] = 0.0
        
        # 右边界：出流 (∂c/∂x=0)
        c[t,N+1,:] = c[t,N,:]
        
        # 周期性边界
        c[t,:,0] = c[t,:,N]    # 下虚拟 = 上物理
        c[t,:,N+1] = c[t,:,1]  # 上虚拟 = 下物理
    
    return c[:, 0:N+1, 0:N+1]  # 返回物理区域

# 将c的结果可视化，需要可视化的时间为t = 0.2,0.5,1.0,1.5
def visualize(c,dt):
    times = [0.2,0.5,1.0,1.5]
    for time_point in times:
        t_index = int(time_point/dt)
        plt.figure()
        plt.imshow(c[t_index].T, extent=[0, 1, 0, 1], origin='lower', cmap='hot', interpolation='nearest', aspect='equal')
        plt.colorbar(label='FVM c(t,x,y)')
        plt.title(f'FVM t={time_point} dt={dt}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'FVM t={time_point} dt={dt}.png',dpi=300)
        plt.close()
        print(f'Visualization for t={time_point} saved as FVM t={time_point} dt={dt}.png')

# main
if __name__ == "__main__":
    h = 0.005
    T = 1.5
    dt = 0.0005
    D = 0.01
    c = FVM(h,T,dt,D)
    visualize(c,dt)
    
    Nsnap = 100
    N = int(1/h)
    unit = int((T/Nsnap)/dt)
    csnap = np.zeros((Nsnap+1,(N+1)*(N+1)))
    for t in range(Nsnap+1):
        t_idx = t*unit
        csnap[t] = c[t_idx].reshape(-1)
    np.savetxt('csnap.txt', csnap)
    
    t_end = 1.5
    n = 3000
    N = int(1/h)
    unit = int((t_end/n)/dt)
    data = np.zeros((n+1,(N+1)*(N+1)))
    for t in range(n+1):
        t_idx = t*unit
        data[t] = c[t_idx].reshape(-1)
    np.savetxt('data.txt', data)
    
    
