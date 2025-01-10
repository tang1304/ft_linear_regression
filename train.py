import csv
import json
import numpy as np
import matplotlib.pyplot as plt


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


def unstandardize(x, y, theta):
	t0 = (y.std() / x.std()) * theta[0][0]
	t1 = y.std() * theta[1][0] - (y.std() * theta[0][0] * x.mean() / x.std()) + y.mean()

	return t0, t1


def model_matrix(x, theta):
	return np.dot(x, theta)


def cost_function(x, y, theta, m):
	return 1/(2*m) * np.sum((model_matrix(x, theta) - y)**2)


def gradients(x, y, theta, m):
	return 1/m * x.T.dot(model_matrix(x, theta) - y)


def gradient_descent_algo(x, y, theta, m, learning_rate, n):
	costs = []
	for i in range(n):
		grad = gradients(x, y, theta, m)
		theta -= learning_rate * grad
		cost = cost_function(x, y, theta, m)
		costs.append(cost)

	return theta, costs


def plotting_first(x_val, y_val):
	plt.scatter(x_val, y_val)
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (€)")
	plt.title("Scatter plot of cars prices according to their mileage")
	plt.show()


def plotting_lg(x_val, y_val, t0, t1):
	lg = t0 * x_val + t1
	plt.scatter(x_val, y_val)
	plt.plot(x_val, lg, color='red')
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (€)")
	plt.title("Scatter plot of cars prices according to their mileage, with the linear regression")
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
		x_stand = (x - x_mean)/x_std
		y_stand = (y - y_mean)/y_std
	else:
		raise ZeroDivisionError()
	print(f"Normalized x: {x_stand}")
	print(f"Normalized y: {y_stand}")
	ones_column = np.ones((m, 1))
	x_stand = np.reshape(x_stand, (m, 1))
	Y = np.reshape(y_stand, (m, 1))
	X = np.hstack((x_stand, ones_column)) # X is the matrix of features
	theta = np.zeros((2, 1)) # theta is the matrix of the thetas (0 at the beginning)
	return x, y, X, Y, theta, m


def main():
	try:
		datas = load_data('data.csv')
		x, y, X, Y, theta, m = process_data(datas)
		thetas, costs = gradient_descent_algo(X, Y, theta, m, 0.001, 2000)
		t0, t1 = unstandardize(x, y, thetas)
		plotting_lg(x, y, t0, t1)
		print(f"theta0 = {t0}, theta1 = {t1}")

		with open("train_data.json", "w") as file:
			json.dump({"theta0": t0, "theta1": t1}, file)

	except Exception as e:
		print(f"Error : {e}")
		return


if __name__ == "__main__":
	main()
