import csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd

def is_number(value):
	try:
		float(value)
		return True
	except ValueError as e:
		print(f"Error : ", e)
		return False


def load_data(file):
	datas=[]
	with open(file, 'r') as csv_file: # 'with' closes automatically the file at the end
		reader = csv.DictReader(csv_file, delimiter=',') # dictReader uses the first row as keys for the dict
		for row in reader:
			if all(is_number(value) for value in row.values()):
				datas.append([float(value) for value in row.values()])
			else:
				print(f"This row will not be used: ", row)
	if not datas:
		raise ValueError("There is no usable data in your file.")
	return datas


def model_matrix(x, theta):
	return np.dot(x, theta)


def cost_function(x, y, theta, m):
	return 1/(2*m) * np.sum((model_matrix(x, theta) - y)**2)


def gradients(x, y, theta, m):
	# print(f"X: {x}, shape: {x.shape}")
	# print(f"Y: {y}, shape: {y.shape}")
	return 1/m * x.T.dot(model_matrix(x, theta) - y)


def gradient_descent_algo(x, y, theta, m, learning_rate, n):
	costs = []
	for i in range(n):
		grad = gradients(x, y, theta, m)
		theta -= learning_rate * grad
		cost = cost_function(x, y, theta, m)
		costs.append(cost)

		if i % 100 == 0:
			print(f"Iteration {i} : cost = {cost}")
			print(f"theta = {theta}")
	return theta, costs


def plotting_first(x_val, y_val):
	plt.scatter(x_val, y_val)
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (€)")
	plt.title("Scatter plot of cars prices according to their mileage")
	plt.show()


def process_data(datas):
	datas = np.array(datas) # to pass from list of lists to array
	x = datas[:, 0]
	y = datas[:, 1]
	# plotting_first(x, y)
	m = len(x) # m is the number of elements

	x_std = np.std(x)
	y_std = np.std(y)
	x_mean = np.mean(x)
	y_mean = np.mean(y)

	if x_std != 0 and y_std != 0:
		x_norm = (x - x_mean)/x_std
		y_norm = (y - y_mean)/y_std
	else:
		raise ZeroDivisionError()
	print(f"Normalized x: {x_norm}")
	print(f"Normalized y: {y_norm}")
	ones_column = np.ones((m, 1))
	x_norm = np.reshape(x_norm, (m, 1))
	y_norm = np.reshape(y_norm, (m, 1))
	X = np.hstack((x_norm, ones_column)) # X is the matrix of features
	Y = np.hstack((y_norm, 1)) # Y is the matrix of labels
	theta = np.zeros((2, 1)) # theta is the matrix of the thetas (0 at the beginning)
	print(f"X: {X}")
	return X, y_norm, theta, m


def main():
	try:
		datas = load_data('data.csv')
		X, Y, theta, m = process_data(datas)
		theta, costs = gradient_descent_algo(X, Y, theta, m, 0.01, 2000)


	except Exception as e:
		print(f"Error : {e}")
		return


if __name__ == "__main__":
	main()