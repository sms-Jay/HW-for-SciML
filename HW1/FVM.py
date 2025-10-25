import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

# problem 1:use finite volume method to solve the 2D Convection-Diffusion Equation
# Default: dx = dy = h

def FVM(h,T,dt,D):
    N = int(1/h) 
    M = int(T/dt)
    mu1 = D*dt*N*N
    mu2 = dt*N
    c = np.zeros((M+1,N+1,N+1))
    #[0,M]*[0,N]*[0,N]
    for t in range(1,M+1):
        # i = 0
        for j in range(N+1):
            y = j*h
            if y > 1/3 and y < 2/3:
                c[t][0][j] = 1
            else:
                c[t][0][j] = 0
        # 1 <= i <= N-1
        for i in range(1,N):
            for j in range(N):
                jm = (j-1+N)%N
                c[t][i][j] = c[t-1][i][j] + mu1*(c[t-1][i-1][j]+c[t-1][i+1][j]+c[t-1][i][j+1]+c[t-1][i][jm]-4*c[t-1][i][j]) 
                + mu2*(c[t-1][i-1][j]-c[t-1][i][j])
            c[t][i][N] = c[t][i][0]
        for j in range(N):
            jm = (j-1+N)%N
            c[t][N][j] = c[t-1][N][j] + mu1*(c[t-1][N-1][j]+c[t-1][N][j+1]+c[t-1][N][jm]-3*c[t-1][N][j]) 
            + mu2*(c[t-1][N-1][j]-c[t-1][N][j])
        c[t][N][N] = c[t][N][0]
    return c

def mat_vec(h,ct):
    N = int (1/h)
    Ct = np.zeros((N+1)*(N+1))
    for i in range(N+1):
        for j in range(N+1):
            idx = i*(N+1)+j
            Ct[idx] = ct[i][j]
    return Ct
# 将c的结果可视化，需要可视化的时间为t = 0.2,0.5,1.0,1.5
def visualize(c,h,dt):
    N = int(1.0/h)
    times = [0.2, 0.5, 1.0, 1.5]
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
    visualize(c,h,dt)
    
    Nsnap = 100
    N = int(1/h)
    unit = int((T/Nsnap)/dt)
    csnap = np.zeros((Nsnap+1,(N+1)*(N+1)))
    for t in range(Nsnap+1):
        t_idx = t*unit
        csnap[t] = mat_vec(h,c[t_idx])
    np.savetxt('csnap.txt', csnap)
