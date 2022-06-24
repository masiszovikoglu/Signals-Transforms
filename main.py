from math import sin
from pydoc import doc
import numpy as np
from excel import get_data
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy.signal import find_peaks


# def get_n_highest_peaks(arr):
#     peaks = []
#     i = 0
#     l = len(arr)
#     while True:
#         if i + 1 == l or arr[i+1] < arr[i]:
#             break
#         else:
#             i += 1
#     peaks.append(arr[i])
#     last = arr[i]
#     new = False
#     for j in range(i, l):
#         if not new and arr[j] < 0.8 * last:
#             new = True
#         if new and j + 1 < l and arr[j + 1] < arr[j]:
#             peaks.append(arr[j])
#             last = arr[j]
#             new = False
#     return peaks

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
    
   
    Tepoch = 0
    my_dict = get_data()
    f = []
    for i in my_dict["Value"]:
        f.append(int(i))
        Tepoch += 10    
    N = len(f)
    Ts = Tepoch / N 
    t = np.arange(0, Tepoch, Ts) 
    M = int(2**(np.ceil(np.log(N)/np.log(2))+4))
    w = (np.arange((M//2)+1)*6*24)/M
    
    fhat = fft(f,M)*Ts
    fhat = fhat[0:(M//2+1)]
    peaks, _ = find_peaks(abs(fhat)[0:len(fhat)//3])
    peaksxy = []
    print("A")
    for p in peaks:
        peaksxy.append((p, abs(fhat)[p]))        
    print("B")
    peaksxy = sorted(peaksxy, key=lambda x: x[1], reverse=True)[0:10]

    plt.plot(w, abs(fhat))
    plt.scatter(list(x[0]*(6*24)/M for x in peaksxy), list(y[1] for y in peaksxy), color="red") 
    plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
    plt.ylabel('unit: $[f][t]$')
    plt.grid()
    #plt.savefig('code1p1.pdf') # save it as pdf
    plt.show() 

    plt.plot(t / 60, f)
    plt.xlabel('time in hours')
    plt.ylabel('height in cm')
    plt.grid()
    plt.show()
    
    # l = len(t) / 50
    # for i in range(50):
    #     plt.plot(t[int(l*i):int(l*(i+1))], f[int(l*i):int(l*(i+1))])
    #     plt.grid()
    #     plt.show()

    # print(len(x))
    
    # plt.plot(np.arange(0,500,1), x[0:500])
    # plt.show()
    
    # fs = 6*24
    # T = 1/fs
    # N = len(f)
    # yf = fft(f)
    # xf = np.linspace(0.0, fs / 2, N//2)
    
    # plt.plot(xf, 2/N * np.abs(yf[:N//2]))
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()