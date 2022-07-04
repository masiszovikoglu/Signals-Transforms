import numpy as np
from excel import get_data
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy.signal import find_peaks

def main():  
    Tepoch = 0
    my_dict = get_data()
    f = []
    for i in range(len(my_dict["Value"])):
        #f.append(int(i))
        f.append(int(100 * (np.sin(2 * np.pi * (i / 144))))) # + np.sin(2 * np.pi * 3 * (i / 144)) + np.sin(2 * np.pi * 5 * (i / 144)) + np.sin(2 * np.pi * (1/6) * (i / 144)) + np.sin(2 * np.pi * (1/25) * (i / 144))
        Tepoch += 1 / 144   
    N = len(f)
    P = 1 #5 #20

    Ts = Tepoch / N 
    t = np.arange(0, Tepoch, Ts) 
    M = int(2**(np.ceil(np.log(N)/np.log(2))+6))
    w = 2*np.pi*(np.arange((M//2)+1))/Ts/M
    
    fhat = fft(f,M)*Ts
    fhat = fhat[0:(M//2+1)]
    peaks, _ = find_peaks(abs(fhat)[0:len(fhat)//3], distance=1000, height=500)
    #print(len(peaks))
    peaksxy = []
    for p in peaks:
        peaksxy.append(((p)/Ts/M, abs(fhat)[p])) 
    peaksxy = sorted(peaksxy, key=lambda x: x[1], reverse=True)[0:P]
    print(peaksxy)

    # plt.plot(t, f)
    # plt.xlabel('time in days')
    # plt.ylabel('height in cm')
    # plt.grid()
    # plt.show()

    # plt.plot(w/(2*np.pi), abs(fhat))
    # plt.scatter(list(x[0] for x in peaksxy), list(y[1] for y in peaksxy), color="red") 
    # plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
    # plt.ylabel('unit: $[f][t]$')
    # plt.grid()
    # #plt.savefig('code1p1.pdf') # save it as pdf
    # plt.show() 
    
    N = 288 #len(f) # number of data points
    b = np.ndarray((N,1), dtype=int)
    
    for i in range (N):
        b[i][0]=f[i]
    #print(b)

    A = np.zeros((N, 2*P)) #, dtype=np.complex_
    for i in range(N):
        for k in range(P):
            #A[i][k] = np.e ** (1j*(2*np.pi) * (peaksxy[k][0]) * i)
            A[i][2*k] = np.cos(2*(np.pi)*peaksxy[k][0] * i)
            A[i][2*k+1] = np.sin(2*(np.pi)*peaksxy[k][0] * i)

    A_star = A.T #A.conj().T
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_star, A)), A_star), b)
    print("x:",X)
    for i in range(10):    
        print("Ax:",np.matmul(A,X)[i], "b:",b[i])
    
    new_samples = []
    for i in range(len(f)):
        value = 0
        for j in range (P):
            value += float(X[int(2*j)][0])*np.cos(2*(np.pi)*float(peaksxy[j][0])*i)+float(X[int(2*j + 1)][0])*np.sin(2*(np.pi)*float(peaksxy[j][0])*i)
            #value += X[j][0] * np.e ** (1j*(2*np.pi) * (peaksxy[j][0]) * i)
        new_samples.append(value)
        
    t = np.arange(0, Tepoch, Ts) 
    # for i in range(10):
    #     print(f[i+1000], new_samples[i+1000])
    plt.plot(t, f)
    plt.plot(t, new_samples, color="red")
    plt.show()
    

if __name__ == "__main__":
    main()