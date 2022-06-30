import numpy as np
from excel import get_data
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy.signal import find_peaks

def main():  
    Tepoch = 0
    my_dict = get_data()
    f = []
    for i in my_dict["Value"]:
        f.append(int(i))
        Tepoch += 1 / 144   
    N = len(f)
    P = 5

    Ts = Tepoch / N 
    t = np.arange(0, Tepoch, Ts) 
    M = int(2**(np.ceil(np.log(N)/np.log(2))+4))
    w = 2*np.pi*(np.arange((M//2)+1))/Ts/M
    
    fhat = fft(f,M)*Ts
    fhat = fhat[0:(M//2+1)]
    peaks, _ = find_peaks(abs(fhat)[0:len(fhat)//3], distance=200, height=1000)
    peaksxy = []
    for p in peaks:
        peaksxy.append(((p)/Ts/M, abs(fhat)[p])) 
    peaksxy = sorted(peaksxy, key=lambda x: x[1], reverse=True)[0:P]
    print(peaksxy)

    plt.plot(w/(2*np.pi), abs(fhat))
    plt.scatter(list(x[0] for x in peaksxy), list(y[1] for y in peaksxy), color="red") 
    plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
    plt.ylabel('unit: $[f][t]$')
    plt.grid()
    #plt.savefig('code1p1.pdf') # save it as pdf
    plt.show() 

    plt.plot(t, f)
    plt.xlabel('time in days')
    plt.ylabel('height in cm')
    plt.grid()
    plt.show()
    
    N = 5 #len(f) # number of data points
    b = np.ndarray((N,1), dtype=int)
    
    for i in range (N):
        b[i][0]=f[i]
    #print(b)

    A = np.zeros((N, P), dtype=np.complex_)
    for i in range(N):
        for k in range(P):
            A[i][k] = np.e ** (1j*(2*np.pi) * (peaksxy[k][0]) * i)

    A_star = A.conj().T
    print("1",np.matmul(A_star, A))
    print("2",np.linalg.inv(np.matmul(A_star, A)))
    print("3",np.matmul(np.linalg.inv(np.matmul(A_star, A)), A_star))
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_star, A)), A_star), b)
    print(X)
    print(np.matmul(A,X), b)
    
    new_samples = []
    for i in range(len(f)):
        value = 0
        for j in range (P):
            #value += np.real(X[j][0])*np.cos(2*(np.pi)*peaksxy[j][0]*t)+np.imag(X[j][0])*np.sin(2*(np.pi)*peaksxy[j][0]*t)
            value += X[j][0] * np.e ** (1j*(2*np.pi) * (peaksxy[j][0]) * i)
        new_samples.append(value)
        
    t = np.arange(0, Tepoch, Ts) 
    plt.plot(t, f)
    plt.plot(t, new_samples, color="red")
    plt.show()
    

if __name__ == "__main__":
    main()