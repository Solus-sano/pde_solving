import numpy as np
import math as mth
import scipy as sp
import matplotlib.pyplot as plt

def init_U(x):
    """U的精确解"""
    return np.exp(x)*np.sin(x)

def f(x):
    return np.exp(x)*(np.sin(x)-2*np.cos(x))

def q(x):
    return 1

def solving(f,q,a,b,alphe,beta,N,flag=1):
    """给定方程及边界条件、步长，用有限差分法求解网函数"""
    G=np.zeros((N+1,N+1))
    B=np.zeros((N+1,1))
    X=np.linspace(a,b,N+1)
    h=(b-a)/N

    for i in range(1,N):
        G[i][i-1:i+2]=np.array([-1/(h*h),2/(h*h)+q(X[i]),-1/(h*h)])
        B[i]=f(X[i])
    
    # G[0][:2]=np.array([2/(h*h)+q(X[1]),-1/(h*h)])
    # G[N-2][-2:]=np.array([-1/(h*h),2/(h*h)+q(X[N-1])])

    # B[0],B[N-2]=f(X[1])+alphe/(h*h),f(X[N-1])+beta/(h*h)
    if flag==1:
        G[0][0], G[N][N]=1, 1
        B[0], B[N]=alphe, beta

    elif flag==2:
        G[0][0:2], G[N][-2:]=np.array([2/(h*h)+q(X[0]),-2/(h*h)]), np.array([-2/(h*h),2/(h*h)+q(X[N])])
        B[0], B[N]=f(X[0])-2*alphe/h, f(X[N])+2*beta/h

    U_lst=np.linalg.inv(G)@B
    # U_lst=np.hstack((np.array([alphe]),U_lst.T[0],np.array([beta])))
    return U_lst.T[0]

def loss_f(U_lst,init_U_lst):
    """计算两网函数的无穷范数"""
    return np.max(np.abs(U_lst-init_U_lst))

def show_solving(a,b,alphe,beta,N=40,flag=1):
    """给定步长，可视化求解"""
    # a,b=0,mth.pi

    x_lst=np.linspace(a,b,N+1)
    y_init=init_U(np.linspace(a,b,500))
    y_solve=solving(f,q,a,b,alphe=alphe,beta=beta,N=N,flag=flag)
    
    plt.plot(np.linspace(a,b,500),y_init)
    plt.scatter(x_lst,y_solve,c='red',s=20)
    plt.title("result")

def show_loss(a,b,alphe,beta,flag=1):
    a,b=0,mth.pi
    n_lst=np.arange(10,100,5)
    e_lst=[]
    for n in n_lst:
        x_lst=np.linspace(a,b,n+1)
        y_init=init_U(x_lst)
        y_solve=solving(f,q,a,b,alphe=alphe,beta=beta,N=n,flag=flag)
        # e_lst.append((loss_f(y_solve,y_init))/((b-a)/n)**2)
        e_lst.append((loss_f(y_solve,y_init)))
        print("N = %d solving finished"%(n))
    print(e_lst)
    plt.figure()
    plt.scatter(n_lst,e_lst,s=20)
    plt.title("E_loss")

def main1():
    """第一类边界条件"""
    show_solving(0,mth.pi,0,0,10,flag=1)
    show_loss(0,mth.pi,0,0,flag=1)

def main2():
    """第二类边界条件"""
    show_solving(0,mth.pi,1,-mth.exp(mth.pi),40,flag=2)
    show_loss(0,mth.pi,1,-mth.exp(mth.pi),flag=2)

if __name__ == '__main__':
    main1()
    plt.show()