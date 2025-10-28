import numpy as np
import cmath
import time
import os
import matplotlib.pyplot as plt
import scipy 

# problem 3:use dynamic model decomposition to solve the 2D Convection-Diffusion Equation
# Default: dx = dy = h

# 步骤： 选择数据X、Y ，建模Y = KX + f； 对(X,1)做SVD以得到\tilde{K}的谱，最后模拟Um

def DMD(n):
    data = np.loadtxt('csnap.txt')
    data = np.c_[data, np.ones(data.shape[0])]
    data = data[:(n+1),:].T
    X = data[:,:-1]
    Y = data[:,1:]
    U, Sigma, Vt = scipy.linalg.svd(X,full_matrices=False)
    # 截断SVD
    Sigma = np.diag(Sigma)
    r = np.linalg.matrix_rank(Sigma)
    U = U[:, :r]
    Sigma = Sigma[:r,:r]   
    Vt = Vt[:r, :]
    Sigma_inv = scipy.linalg.inv(Sigma)
    K_Tilde = U.conj().T @ Y @ Vt.conj().T @ Sigma_inv
    eigenvalues, left_ev, right_ev = scipy.linalg.eig(K_Tilde, left=True, right=True)
    plot_dmd_eigenvalues(eigenvalues, title="DMD Eigenvalues")
    left_ev = (U @ left_ev).conj().T
    right_ev = Y @ Vt.conj().T @ Sigma_inv @ right_ev
    for i in range (r):
        ip = np.dot(left_ev[i,:],right_ev[:,i])
        factor = 1.0/cmath.sqrt(ip)
        right_ev[:,i] = right_ev[:,i]*factor
        left_ev[i,:] = left_ev[i,:]*factor.conjugate()
    return eigenvalues, left_ev, right_ev


    
def plot_dmd_eigenvalues(z, title="DMD Eigenvalues"):
    """
    在复平面上绘制复数向量的散点图
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 提取实部和虚部
    real_part = z.real
    imag_part = z.imag
    
    # 绘制散点
    scatter = ax.scatter(real_part, imag_part, c=range(len(z)), 
                        cmap='viridis', alpha=0.7, s=50)
    
    # 添加颜色条
    plt.colorbar(scatter, label='Index')
    
    # 设置坐标轴
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(title)
    # 添加原点
    ax.plot(0, 0, 'ro', markersize=8)
    
    plt.savefig(f'DMD Eigenvalues',dpi=300)
    plt.close()

def solve(h,T,dt,n):
    N = int(1/h)
    M = int(T/dt)
    total_num = (N+1)*(N+1)
    

    c = np.zeros((M+1,total_num+1),dtype = complex)
    c[0][total_num] = 1

    eigenvalues, left_ev, right_ev = DMD(n)
    r = left_ev.shape[0]
    for m in range(1,M+1):
        for j in range(r):
            b_j = np.vdot(left_ev[j,:], c[0])  # 先计算内积（标量）
            c[m] += np.pow(eigenvalues[j],m) * b_j * right_ev[:,j]
    
    C = np.zeros((M+1,N+1,N+1))
    for m in range(M+1):
        C[m] = c[m][:-1].real.reshape(N+1,N+1)
    return C

def visualize(c,dt):
    times = [0.2, 0.5, 1.0, 1.5,2.0,3.0,4.0,4.5]
    for time_point in times:
        t_index = int(time_point/dt)
        plt.figure()
        plt.imshow(c[t_index].T, extent=[0, 1, 0, 1], origin='lower', cmap='hot', interpolation='nearest', aspect='equal')
        plt.colorbar(label='DMD c(t,x,y)')
        plt.title(f'DMD t={time_point} dt={dt}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'DMD t={time_point} dt={dt}.png',dpi=300)
        plt.close()
        print(f'Visualization t={time_point} saved as DMD t={time_point} dt={dt}.png')
            
if __name__ == "__main__":
    h = 0.005
    T = 4.5
    D = 0.01
    n = 100
    t_end = 1.5
    dt = t_end/n
    dt = 0.015
    c = solve(h,T,dt,n)
    visualize(c,dt)
    