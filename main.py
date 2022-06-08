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

def show_graph(x, y):
    plt.plot(x, y)
    plt.show()

def main():
    #my_dict = get_data()
    my_dict = {}
    my_dict["Value"] = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]
    #print(my_dict)
    #show_graph(get_x(my_dict["Value"]), my_dict["Value"])
   
    sr = 54
    ts = 1/sr
    t = np.arange(0,1,ts)
    x = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]
    
    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T 

    plt.stem(freq, np.abs(X))
    #plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.show()

    plt.plot(t, ifft(X))
    plt.show()

if __name__ == "__main__":
    main()