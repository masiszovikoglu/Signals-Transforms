import numpy as np
from excel import get_data
from excel import get_data_now
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy.signal import find_peaks
import pandas as pd


def main():  
    Tepoch = 0
    my_dict = get_data()
    f = []
    for i in my_dict["Value"]:
        f.append(int(i))
        # val = np.sin(2 * np.pi * (i / 144)) + np.sin(2 * np.pi * 3 * (i / 144)) + np.sin(2 * np.pi * 5 * (i / 144)) + np.sin(2 * np.pi * (1/6) * (i / 144)) + np.sin(2 * np.pi * (1/25) * (i / 144))
        # f.append(int(100 * val)) 
        Tepoch += 1 / 144   
    N = len(f)
    P = 7

    Ts = Tepoch / N 
    t = np.arange(0, Tepoch, Ts) 
    M = int(2**(np.ceil(np.log(N)/np.log(2))+6))
    w = 2*np.pi*(np.arange((M//2)+1))/Ts/M
    
    fhat = fft(f,M)*Ts
    fhat = fhat[0:(M//2+1)]
    peaks, _ = find_peaks(abs(fhat)[0:len(fhat)//3], distance=4000, height=500)
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

    plt.plot(w/(2*np.pi), abs(fhat))
    plt.scatter(list(x[0] for x in peaksxy), list(y[1] for y in peaksxy), color="red") 
    plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
    plt.ylabel('unit: $[f][t]$')
    plt.grid()
    # #plt.savefig('code1p1.pdf') # save it as pdf
    plt.show() 
    
    N = 4000 #len(f) # number of data points
    b = np.ndarray((N,1), dtype=int)
    
    for i in range (N):
        b[i][0]=f[i]
    #print(b)

    A = np.zeros((N, 2*P)) #, dtype=np.complex_
    for i in range(N):
        for k in range(P):
            #A[i][k] = np.e ** (1j*(2*np.pi) * (peaksxy[k][0]) * i)
            A[i][2*k] = np.cos(2*(np.pi)*peaksxy[k][0] * (i / 144))
            A[i][2*k+1] = np.sin(2*(np.pi)*peaksxy[k][0] * (i / 144))

    A_star = A.T #A.conj().T
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_star, A)), A_star), b)
    #print("x:",X)
    new_samples = []
    phase=-0.0015
    
    
    for i in range(2*len(f)):
        value = 0
        for j in range (P):
            value += float(X[int(2*j)][0])*np.cos((2*(np.pi)+phase)*float(peaksxy[j][0])*(i/144))+float(X[int(2*j +1)][0])*np.sin((2*(np.pi)+phase)*float(peaksxy[j][0])*(i/144))
            value=value+3.5
            #value += X[j][0] * np.e ** (1j*(2*np.pi) * (peaksxy[j][0]) * i)
        new_samples.append(value)
        
    t = np.arange(0, Tepoch, Ts)
    tf = np.arange(0, 2*Tepoch, Ts)
    # for i in range(10):
    #     print(f[i+1000], new_samples[i+1000])
    plt.plot(t, f)
    #plt.plot(tf, new_samples, color="red")
    plt.show()
    start = pd.Timestamp('2021-05-24')
    end = pd.Timestamp('2022-05-24')
    #t = np.linspace(start.value, end.value, len(f))
    #t = pd.to_datetime(t)
    #plt.plot(t, f)
    #plt.plot(t, new_samples, color="red")
    #plt.show()
    
    error = []
    error2 =[]
    success= 0
    miss=0
    wrong_pass=0
    
    count=0
    count_newdata=1
    for i in range (len(f)):
        if(f[i]>150):
            count=count+1
        if(f[i]>150 and new_samples[i+1]>155):
            success=success+1
        if(f[i]>150 and   new_samples[i+1]<155):
            miss=miss+1
        if( f[i]<150 and   new_samples[i+1]>155):
            wrong_pass=wrong_pass+1
            
        
        if(f[i]<150 and f[i]>-150):
         e= abs(f[i]-new_samples[i])
         e2= (f[i]-new_samples[i])
         error.append(e)
         error2.append(e2)
    print(np.mean(error))
    print(np.mean(error2))
    print(my_dict["Date"][7920])
    
    seven_jul=np.array(f[6336:6480])
    eighteen_jul = np.array(f[7920:8064])
    for i in range(16560, 16704):
        if f[i]>150:
            print(my_dict["Date"][i])

    print("approximation:")
    for i in range(16560, 16704):
        if new_samples[i]>150:
            print(my_dict["Date"][i])
            
    
    
    print(success/count)
    print(miss/count)
    print(wrong_pass/(len(f)-count))
    
    
    tt=np.linspace(0,1,144)
    year=52560
    plt.plot(tt, seven_jul)
    plt.plot(tt, new_samples[6336:6480], color="red")
    plt.axhline(y=150, color='green', linestyle='-')
    plt.show()
    my_dict2=get_data_now()
    june=my_dict2["Value"]
    K=len(june)
    print(my_dict["Date"][1600])
    start = pd.Timestamp('2022-06-04')
    end = pd.Timestamp('2022-07-04 19:20')
    t = np.linspace(start.value, end.value, K)
    t = pd.to_datetime(t)
    tt3=t
    plt.plot(tt3, june)
    plt.plot(tt3, new_samples[1600+year:1600+K+year], color="red")
    plt.axhline(y=150, color='green', linestyle='-')
    plt.show()
    
    error = []
    error2 =[]
    success= 0
    miss=0
    wrong_pass=0
    
    count=0
    count_newdata=1
    for i in range (len(june)):
        if(june[i]>150):
            count=count+1
        if(june[i]>150 and new_samples[1600+year+i]>155):
            success=success+1
        if(june[i]>150 and   new_samples[1600+year+i]<155):
            miss=miss+1
        if( june[i]<150 and   new_samples[1600+year+i]>155):
            wrong_pass=wrong_pass+1
            
            
            

    
    print(success/count)
    print(miss/count)
    print(wrong_pass/(len(f)-count))
    
    print(my_dict["Date"][6192])
    day=144
    #tt4=np.linspace(0,4,4*day)
    
    start = pd.Timestamp('2022-07-06')
    end = pd.Timestamp('2022-07-10')
    t = np.linspace(start.value, end.value, 4*day)
    t = pd.to_datetime(t)
    tt4=t
    #plt.plot(tt3, june)
    plt.plot(tt4, new_samples[6192+year:6192+year+4*day], color="red")
    plt.axhline(y=150, color='green', linestyle='-')
    plt.show()
    
    start = pd.Timestamp('2022-07-17')
    end = pd.Timestamp('2022-07-20')
    t = np.linspace(start.value, end.value, 3*day)
    t = pd.to_datetime(t)
    tt4=t
    #plt.plot(tt3, june)
    plt.plot(tt4, new_samples[7920+year:7920+year+3*day], color="red")
    plt.axhline(y=150, color='green', linestyle='-')
    plt.show()
    
    print("Prediction:")
    for i in range(6192+year-144, 6192+year+4*day-144):
        if new_samples[i]>150:
            print(my_dict["Date"][i-year])

    
    

if __name__ == "__main__":
    main()