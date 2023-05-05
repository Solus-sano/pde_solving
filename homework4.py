import numpy as np
import math as mth

def init_U_map(J,t):
    """U的精确解"""
    X = np.linspace(0,1,J+1)
    U_map = mth.cos(4*mth.pi*t)*np.sin(4*mth.pi*X) + (mth.sin(8*mth.pi*t) * np.sin(8*mth.pi*X) / (8*mth.pi))
    # U_map = np.zeros((N+1,))
    # for i in range(N+1):
    #     U_map[i] = mth.exp()
    return U_map

def forward_solving(N,J,t):
    """前向"""
    h = 1/J; tao = 1/N; r = tao/h
    U_map = np.zeros((3,J+1))
    phi_x = np.sin(4*mth.pi*np.linspace(0,1,J+1))

    U_map[0] = phi_x
    U_map[1][1:-1] = (tao * np.sin(8*mth.pi*np.linspace(0,1,J+1)))[1:-1] + phi_x[1:-1] + 0.5*r**2 * np.diff(phi_x,n=2)
    U_map[1][0] = 0; U_map[1][-1] = 0

    for i in range(2,int(t/tao)+1):
        for j in range(1,J):
            U_map[i%3][j] = r**2 * U_map[(i-1)%3][j-1] + (2-2*r**2)*U_map[(i-1)%3][j] + r**2 * U_map[(i-1)%3][j+1] - U_map[(i-2)%3][j]
        U_map[i%3][0] = 0; U_map[i%3][J] = 0
        # print(np.max(np.abs(U_map[i%2])))

    return U_map[(int(t/tao))%3]


def loss_f(U_lst,init_U_lst):
    """计算两网函数的无穷范数"""
    return np.max(np.abs(U_lst-init_U_lst))

def text(N,J):
    print("----------------------------------------------------")
    print(f"N={N}, J={J}: ")
    for t in range(1,6):
        print(f"t={t}: ",loss_f(init_U_map(J=J,t=t),forward_solving(N=N,J=J,t=t)))


if __name__ == '__main__':
    text(500,400)
    text(400,400)


