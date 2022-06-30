import numpy as np

def adjoint(A):
    A_star = A.transpose()
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i][j] = A[i][j].conj()
    return A_star
        
a = np.array([[1+1j, 2+2j],
       [3+3j, 4-4j],[5, 6+6j]])
a_star = a.conj().T

print(a)
print(adjoint(a))
print(a_star)
