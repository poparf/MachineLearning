import numpy as np
import time

a = np.array([1.0,2.0,3.0])
b = 2.0
# it stretches 2.0 in a list of 3 numbers ( 2.0 )
# and then multiplies
# it doesn t make copies
print(a * b)

# vectors - ordered array of numbers
# same type
# Number of elements in the vector -> dimension/rank

zeros_vector = np.zeros(4)
random_vector = np.random.random_sample(4)
aranged_vector = np.arange(4)

# Operations:
# - indexing
# - slicing

a = np.arange(10)
print(f"a = {a}")

# start:stop:step
print(f"a[2:7:1] = ", a[2:7:1])

print(f"a[::-1] = {a[::-1]}")

print(f"np.sum => {np.sum(a)}")
print(f"np.mean => {np.mean(a)}")

print(f"The dot product multiplies values in two vectors element-wise and then sum the result")
b = np.arange(10)
print(np.dot(a,b))


# Vector vs for loop
np.random.seed(1)

a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(f"np.dot(a,b) = {c:.4f}")
print(f"Vectorized version duration: {1000 * ( toc - tic):.4f}ms")

def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

tic = time.time()  # capture start time
#c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

# So, vectorization provides a large speed up in this example. 
# This is because NumPy makes better use of available data parallelism
# in the underlying hardware. GPU's and modern CPU's implement
# Single Instruction, 
# Multiple Data (SIMD) pipelines allowing multiple operations
# to be issued in parallel. 
# his is critical in Machine Learning where the data sets are 
# often very large.


# Matrices are two dimensional arrays
# Denoted with capital bold letter X
# m is the number of rows
# n the number of columns


mx = np.arange(20).reshape(4,5)
print(mx)

print(f"mx[:, 2:4:1] = \n{mx[:,2:4:1]}")