# MachineLearning0.0
Machine learning Algorithms in Python and Octave codes

NumPy learning and usages: 
Numpy tutorial:

A = np.arange(9).reshape(3,3)


a = np.array([[1,2], 
              [2,3]])
b = np.array([[4,5],
              [6,7]])
              
For 1 dimension vector in Numpy : theta = np.zeros((2,1),  dtype=int)

# Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]
 
# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]
              
              
For linear regression it should be (H(theta) - y).T @ (H(theta) - y)

# Data access in array 
print(A[0]) 
print(A[0, 1]) # Access the second item of the first row
print(A[:, 1]) # Access the second column

# Mathematical operator
You can use the operations '', '*', '', '+' and '-' on numpy arrays and they operate elementwise.
For power print(a**2)
np.sum(a, axis=0) : do column wise sum
np.sum(a, axis=1) : do row wise sum

## Linear Algebra

In this course, we use the numpy arrays for linear algebra.
We usually use 1D arrays to represent vectors and 2D arrays to represent
matrices

Note that taking the transpose of a 1D array has NO effect.

Angle between two vector A, B 
cos(alpha) = (A.T @ B) / math.sqrt( (A.T @ A) * (B.T @ B))

Distance between vector A and B : math.sqrt((A -B).T @ (A - B))
