import math,copy
import numpy as np
import matplotlib.pyplot as plt
# Problem statement
"""
    We get a list of house sizes in m2 and the price for each.
    We want to predict the price of an apartemnt based on the size.
"""

# Features
x_train = np.array([77, 31, 65, 69, 76, 50, 80, 54, 46, 60, 38, 54])
# Targets
y_train = np.array([100, 35, 128, 109, 130, 74, 134, 86, 95, 125, 79, 97])

# function to calculate the cost
# The cost represents the error in the prediction
# x - size of apartaments (features)
# y - price of apartments (targets)
# w,b - weight of the regression equation
def compute_cost(x, y, w, b):
    
    # m is the number of training sets
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b # regression equation
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    
    return total_cost

# You utilize the input training data to fit the parameters w,b by
# minimizing a measure of the error between our predictions
# and the actual data


# dj_db => derivative of function J in raport cu var  b
# w.r.t => with respect to


# Calculating the gradient:

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
     
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    return dj_dw / m, dj_db / m


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Alpha represents the step in the downhill ! Be careful a big step might diverge!
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
    
    return w, b


def predict_price(size):
    w_init = 0
    b_init = 0
    iterations = 10000
    tmp_alpha = 1.0e-4

    w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
    
    predicted_price = w_final * size + b_final
    print(f"{size} m2 apartment prediction: {predicted_price} thousands of euros.")
    plt.scatter(x_train, y_train)
    plt.plot(x_train, w_final*x_train+b_final)
    plt.scatter(size, predicted_price, c='red')
    plt.xlabel('House size ( m2 )')
    plt.ylabel('Price ( thosands of euros )')
    plt.show()

    

predict_price(50)
# 40000