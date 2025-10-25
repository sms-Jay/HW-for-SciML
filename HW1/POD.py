import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

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
    Ur, Sigma, Vr = np.linalg.svd(S.T,full_matrices=False)
    V = Ur[:,:k]
    return V

def F(h,D):
    N = int(1/h)
    mu = D/(h*h)
    f = np.zeros((N+1)*(N+1))
    for j in range(N+1):
        y = j*h
        tep = 0
        if y > 1/3 and y < 2/3:
            tep = 1
        else:
            tep = 0
        c = (N+1)+j
        f[c] = (mu+1/h)*tep
    return f
def A_x(x,h,D):
    N = int (1/h)
    mu = D/(h*h)
    Ax = np.zeros((N+1)*(N+1))
    # i = 0 , 1
    for j in range(N+1):
        Ax[j] = 0
        # i = 1
        c = (N+1)+j
        r = 2*(N+1)+j
        u = (N+1)+(j+1)%N
        d = (N+1)+(j-1+N)%N
        if j == N-1 :
            u = (N+1)+N 
        Ax[c] = mu*(x[r]+x[u]+x[d]-4*x[c]) - N*x[c]
        # 2 <= i <= N-1
    for i in range(2,N):
        for j in range(N+1):
            c = i*(N+1)+j
            l = (i-1)*(N+1)+j
            r = (i+1)*(N+1)+j
            u = i*(N+1)+(j+1)%N
            d = i*(N+1)+(j-1+N)%N
            if j == N-1 :
                u = i*(N+1)+N 
            Ax[c] = mu*(x[l]+x[r]+x[u]+x[d]-4*x[c]) + N*(x[l]-x[c])
    i = N
    for j in range(N+1):
        c = i*(N+1)+j
        l = (i-1)*(N+1)+j
        u = i*(N+1)+(j+1)%N
        d = i*(N+1)+(j-1+N)%N
        if j == N-1 :
            u = i*(N+1)+N 
        Ax[c] = mu*(x[l]+x[u]+x[d]-3*x[c]) + N*(x[l]-x[c])
    return Ax

def solve(h,T,Nsnap,k,dt,D):
    N = int (1/h)
    V = POD(h,T,Nsnap,k)
    M = int (T/dt)
    q = np.zeros((M+1,k))
    c = np.zeros((M+1,(N+1)*(N+1)))
    f = F(h,D)
    for j in range(N+1):
        y = j*h
        if y > 1/3 and y < 2/3:
            c[0][j] = 1
        else :
            c[0][j] = 0
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
        for i in range(N+1):
            for j in range(N+1):
                idx = i*(N+1)+j
                C[m][i][j] = c[m][idx]
    return C

def visualize(c,h,dt,k):
    N = int(1.0/h)
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
    k = 10
    c = solve(h,T,Nsnap,k,dt,D)
    visualize(c,h,dt,k)
    
    
        
            