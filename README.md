from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.config.acl_interfaces.acl_interfaces import Acl_interfaces

# ft_linear_regression
The aim of this project is to introduce us to the basic concept behind machine learning. For this project, we will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

## 1. Getting the datas

First, we get the data form the csv file, that we store in a list of lists

## 2. The model

The linear regression is used to find a linear relationship between a dependant variable (here the mileage of the car) and an independent variable (the price of the car). Since it gives us a line, it will be defined as:
	f(x) = a.x + b

### The cost function (Mean Squared Error)

![alt text](./images/cost_function_squarred_error.png)

The cost function represents the difference between the predicted value and the actual value. The goal is to minimize this difference.
Here, 'm' is the number of samples, 'Sigma' is the sum of the differences between the predicted value and the actual value, squared to avoid negative values, on all the samples.

### The gradient descent algorithm

The result of the cost function is a parabola, and the goal is to find the minimum of this parabola, which corresponds to the minimal error between the predicted value and the actual value.
The gradient descent algorithm is used to find this minimum. It is based on the derivative of the cost function, which gives us the slope of the curve at a given point. The goal is to find the point where the slope is equal to 0, which is the minimum of the curve.

![alt text](./images/gradient_descent_on_mean_squared_error.png)

We will use a random starting point, and we will update the value of 'a' by advancing with a constant step in the direction of the slope. We'll do this until the slope is equal to 0.


## 3. Using matrix calculations

So to avoid multiple calculations, we can use matrix multiplication, which will apply a formula to all the samples at once. This will allow us to calculate the cost function and the gradient descent algorithm much faster.

Our model is now defined as:
	f(X) = X . Theta

Where X is a matrix of the samples, and Theta is a matrix of the parameters 'a' and 'b'. These represent the 'ax + b' part of the cost function.





Reminder about a matrix multiplication:

![alt text](./images/matrix_multiplication.png)


![alt text](./images/cost_function_matrix.png)


## Numpy on matrix

Numpy is a library that allows us to work with arrays and matrices. It is very useful for linear algebra operations, and it is very efficient for large datasets.
To declare a matrix, we can use the following syntax (define a matrix in capital letter):

```python
# matrix 2x3
A = np.array([[1, 2], [3, 4], [5, 6]])

A.shape() # gives the dimensions of the matrix
(2, 3)

A.T # gives the transpose of the matrix
(3, 2)

B = np.ones((3, 1)) # matrix 3x1 filled with 1

np.dot(A, B) # gives the matrix multiplication of A and B
#Note: The A.y factor and the B.x factor must be the same !!!
```

## Ressources

https://www.youtube.com/@MachineLearnia/videos

https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/

https://numpy.org/doc/stable/reference/index.html

https://matplotlib.org/stable/users/index

https://dilipkumar.medium.com/linear-regression-model-using-gradient-descent-algorithm-50267f55c4ac