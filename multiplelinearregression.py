import copy, math
import matplotlib.pyplot as plt
import numpy as np

# You will use the motivating example of housing price prediction. The training dataset contains three examples with four features (size, bedrooms, floors and, age) shown in the table below.  Note that, unlike the earlier labs, size is in sqft rather than 1000 sqft. This causes an issue, which you will solve in the next lab!
# 
# | Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
# | ----------------| ------------------- |----------------- |--------------|-------------- |  
# | 2104            | 5                   | 1                | 45           | 460           |  
# | 1416            | 3                   | 2                | 40           | 232           |  
# | 852             | 2                   | 1                | 35           | 178           |  
# 
# You will build a linear regression model using these values so you can then predict the price for other houses. For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.  

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predict(x, w, b):
    return np.dot(x, w) + b

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    
    return cost / ( 2 * m)

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w: {cost}')

def compute_gradient(X, y,  w, b):
    
    m, n = X.shape # (num of examples, num of features)
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
        
    return dj_db / m, dj_dw / m

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
      
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        
        dj_db, dj_dw, = compute_gradient(X, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w, b

initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7

w_final, b_final = gradient_descent(X_train,
                                    y_train,
                                    initial_w,
                                    initial_b,
                                    alpha,
                                    iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

# now you need a list [] of features
# and then use the next function to predict the values:
features = np.array([2104, 5, 1, 45])
print(predict(features, w_final, b_final))