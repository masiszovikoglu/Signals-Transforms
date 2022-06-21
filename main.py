from math import sin
import numpy as np
from excel import get_data
from matplotlib import pyplot as plt
from numpy.fft import fft

# temporary
def get_x(values):
    x = []
    for i in range(len(values)):
        x.append(i)
    return x

def main():
    #x = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]
    # x = []
    # for i in range(500):
    #     x.append(np.sin(i/10)+np.sin(i/1.5))
    
   
    Tepoch = 1
    my_dict = get_data()
    f = []
    for i in my_dict["Value"]:
        f.append(int(i))
        Tepoch += 10
    Ts = Tepoch / len(f) 
    #Ts = 0.01
    t = np.arange(0, Tepoch, Ts) #+Ts/2
    # f = np.cos(2*np.pi*5*t)+np.cos(2*np.pi*10*t)/2
    N = len(f)
    M = int(2**(np.ceil(np.log(N)/np.log(2))+4))
    w = np.arange((M//2)+1)*2*np.pi/M/Ts
    fhat = fft(f,M)*Ts
    fhat = fhat[0:(M//2+1)]

    plt.plot(w/(2*np.pi), abs(fhat))
    plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
    plt.ylabel('unit: $[f][t]$')
    plt.grid()
    #plt.savefig('code1p1.pdf') # save it as pdf
    plt.show() 
    l = len(t) / 50
    for i in range(50):
        plt.plot(t[int(l*i):int(l*(i+1))], f[int(l*i):int(l*(i+1))])
        plt.grid()
        plt.show()

    # print(len(x))
    
    # plt.plot(np.arange(0,500,1), x[0:500])
    # plt.show()
    
    # fs = 6*24
    # T = 1/fs
    # N = len(x)
    # yf = fft(x)
    # xf = np.linspace(0.0, fs / 2, N//2)
    
    # plt.plot(xf, 2/N * np.abs(yf[:N//2]))
    # plt.show()


if __name__ == "__main__":
    main()