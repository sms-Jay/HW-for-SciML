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
    data = np.loadtxt('data.txt')
    csnap = np.c_[data, np.ones(data.shape[0])]
    csnap = csnap[:(n+1),:].T
    X = csnap[:,:-1]
    Y = csnap[:,1:]
    U, Sigma, Vt = scipy.linalg.svd(X,full_matrices=False)
    # 截断SVD
    Sigma = np.diag(Sigma)
    r = np.linalg.matrix_rank(Sigma)
    U = U[:, :r]
    Sigma = Sigma[:r,:r]
    Vt = Vt[:r, :]
    Sigma_inv = scipy.linalg.inv(Sigma)
    K_Tilde = U.T @ Y @ Vt.T @ Sigma_inv
    eigenvalues, left_ev, right_ev = scipy.linalg.eig(K_Tilde, left=True, right=True)
    # print(eigenvalues)
    left_ev = left_ev.T @ U.T 
    right_ev = Y @ Vt.T @ Sigma_inv @ right_ev
    for i in range (r):
        ip = np.dot(left_ev[i,:],right_ev[:,i])
        factor = cmath.sqrt(ip)
        right_ev[:,i] = right_ev[:,i]/factor
        left_ev[i,:] = left_ev[i,:]/factor
    print(left_ev @ right_ev)
    return eigenvalues, left_ev, right_ev
def solve(h,T,dt,n):
    N = int(1/h)
    M = int(T/dt)
    total_num = (N+1)*(N+1)
    c = np.zeros((M+1,total_num+1),dtype = complex)
    c[0][total_num] = 1
    eigenvalues, left_ev, right_ev = DMD(n)
    # eigenvalues = np.real(eigenvalues)
    # left_ev = np.real(left_ev)
    # right_ev = np.real(right_ev)
    r = left_ev.shape[0]
    for m in range(1,M+1):
        for j in range(r):
            b_j = np.dot(left_ev[j,:], c[0])  # 先计算内积（标量）
            c[m] += np.pow(eigenvalues[j],m) * b_j * right_ev[:,j]
    
    C = np.zeros((M+1,N+1,N+1))
    for m in range(M+1):
        for i in range(N+1):
            for j in range(N+1):
                idx = i*(N+1)+j
                C[m][i][j] = np.real(c[m][idx])
    return C

def visualize(c,dt):
    times = [0.2, 0.5, 1.0, 1.5]
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
    T = 1.5
    dt = 0.0005
    D = 0.01
    n = 100
    c = solve(h,T,dt,n)
    visualize(c,dt)
    