import numpy as np
import pandas as pd 
import numpy as np
import math
import numba as nb
import random
import csv
import random
import statistics 
import sys
from numpy.linalg import inv

alpha = float(sys.argv[1]) #fee cjefficient 1
beta = float(sys.argv[2]) #fee coefficient 2
gamma=float(sys.argv[3])

@nb.njit(parallel=True)
def matrix_mult(A, B):
     m, n = A.shape
     _, p = B.shape
     C = np.zeros((m, p))
     for j in range(n):
       for k in range(p):
         for i in range(m):
           C[i, k] += A[i, j] * B[j, k]
     return C

table=pd.read_csv('data.csv')
full=table.to_numpy()
omega_1=full[0]
omega_n=full[99]
omega=[]
for i in range (100):
  omega.append(omega_n[i] - omega_1[i])

omega=np.array(omega).reshape(-1,1)
Dm=np.zeros((100,100))
L=np.ones((1,19))


for i in range(0,100):
    Dm[i][i]=random.randrange(340)
    if i<int(math.log(1e6,2)):
       L[0][i]=2**(int(math.log(1e6,2)) - (int(math.log(1e6,2)-i) ))

left_element=beta*Dm.dot(omega.dot(omega.transpose().dot(Dm)))
ExI=alpha*np.ones(left_element.shape)
np.fill_diagonal(ExI, 0)

np.add(ExI, left_element)
#dummy.reshape(-1,1)
right_element=beta*matrix_mult(Dm, omega).dot(L)

lower_right_element=beta*matrix_mult(L.transpose(), L)
final = np.hstack((left_element, right_element))
final_2=np.hstack((right_element.transpose(), lower_right_element))
final=np.vstack((final,final_2))
dumm=list(final)
matrix=[]
for i in range(final.shape[0]):
  for j in range(final.shape[1]):
    if (final[i,j]!=0):
      arr_f=([int(i+1), int(j+1), final[i,j]])
      matrix.append(arr_f)

matrix=np.array(matrix)

with open("temp.csv", "w") as f:
  wr=csv.writer(f, delimiter = " ")
  wr.writerows(matrix)
