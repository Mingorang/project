#imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


import numpy as np
from scipy.linalg import solve
#Determining matrix columns and rows
size = int(input("What is the size of the simultaneous equation? (2-4): "))
print("")
if size == 2:
    print("Matrix A will be a 2 x 1 matrix, so matrix b will have 2 variables x and y.")
elif size == 3:
    print("Matrix A will be a 3 x 1 matrix, so matrix b will have 3 variables x, y and z.")
elif size == 3:
    print("Matrix A will be a 4 x 1 matrix, so matrix b will have 4 variables x, y, z and t.")

#Eqauations are of form Ax=b
A = []
b = []

#Finding matrix A (transformation matrix)
for i in range(size):
    row = []
    print("")
    for j in range(size):
        transform = float(input("Coefficient at specific point: "))
        row.append(transform)
    A.append(row)
A = np.array(A)
print(A)

#Finding matrix b (position vector), same method as matrix A
for k in range(size):
    row_1 = []
    position = float(input("Value at the right of the = sign: "))
    row_1.append(position)
    b.append(row_1)
b = np.array(b)
#Solution using scipy, calculating inverse of A can introduce errors especially when using floats as coefficients
solution = solve(A,b)
#Outputting solutions
if size ==2:
    print(f"X = {solution[0]}")
    print(f"Y = {solution[1]}")
elif size == 3:
    print(f"X = {solution[0]}")
    print(f"Y = {solution[1]}")
    print(f"Z = {solution[2]}")                           
elif size == 4:
    print(f"X = {solution[0]}")
    print(f"Y = {solution[1]}")
    print(f"Z = {solution[2]}")
    print(f"T = {solution[3]}")

