from math import sin
import numpy as np
from excel import get_data
from matplotlib import pyplot as plt
from numpy.fft import fft, ifft

# temporary
def get_x(values):
    x = []
    for i in range(len(values)):
        x.append(i)
    return x

def main():
    #x = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]
    x = []
    for i in range(500):
        x.append(np.sin(i/10)+np.sin(i/1.5))
    
   
    # my_dict = get_data()
    # x = []
    # for i in my_dict["Value"]:
    #     x.append(int(i))

    # sr = len(x)
    # ts = 1/sr
    # t = np.arange(0,1,ts)/2
    #t = np.arange(0, sr, 1)
    print(len(x))
    
    #plt.plot(t[0:1440], x[0:1440])
    plt.plot(np.arange(0,500,1), x[0:500])
    plt.show()
    # plt.plot(t[9000:10000], x[9000:10000])
    # plt.show()
    
    fs = 1
    T = 1/fs
    N = len(x)
    yf = fft(x)
    xf = np.linspace(0.0, fs / 2, N//2)
    
    plt.plot(xf, 2/N * np.abs(yf[:N//2]))
    plt.show()

    # x_new = ifft(X)
    # # for i in range(len(x)):
    # #     if x[i] != x_new[i]:
    # #         print(x[i], x_new[i])
    # plt.plot(t[0:1440], x_new[0:1440])
    # plt.show()

if __name__ == "__main__":
    main()