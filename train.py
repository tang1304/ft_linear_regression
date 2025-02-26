import csv
import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def is_positive_number(value):
	try:
		nbr = float(value)
		if nbr < 0:
			raise ValueError(f"{nbr} is not a positive number.")
		return True
	except ValueError as e:
		print(f"Error : ", e)
		return False


def load_data(file):
	datas=[]
	with open(file, 'r') as csv_file: # 'with' closes automatically the file at the end
		reader = csv.DictReader(csv_file, delimiter=',') # dictReader uses the first row as keys for the dict
		for row in reader:
			if all(is_positive_number(value) for value in row.values()):
				datas.append([float(value) for value in row.values()])
			else:
				print(f"This row will not be used: ", row)
	if not datas:
		raise ValueError("There is no usable data in your file.")
	return datas


def unstandardize(x, y, t0, t1):
	t1 = t1 * y.std() / x.std()
	t0 = y.mean() - (t1 * x.mean())
	return t0, t1


def cost_function(y, estimate_price, m):
	return (1 / (2 * m)) * np.sum((estimate_price - y)**2)


def gradients(x, y, theta, m):
	return 1/m * x.T.dot(np.dot(x, theta) - y)


def gradient_descent_algo(x, y, m, learning_rate, n):
	costs = []
	t0 = 0
	t1 = 0
	for i in range(n):
		estimate_price = t0 + np.dot(x, t1)
		w_derived = (1/m) * np.dot(x, (estimate_price - y))
		b_derived = (1/m) * np.sum(estimate_price - y)
		cost = cost_function(y, estimate_price, m)
		costs.append(cost)
		t0 -= learning_rate * b_derived
		t1 -= learning_rate * w_derived
		if i % 100 == 0:
			print(f"Iteration {i} : cost = {cost} ; theta0 = {t0} ; theta1 = {t1}")
	plt.plot(costs)
	plt.xlabel("Number of iterations")
	plt.ylabel("Cost")
	plt.title("Linear regression cost")
	plt.savefig("Cost.png")
	plt.clf()
	return t0, t1, costs


def plotting_first(x_val, y_val):
	plt.scatter(x_val, y_val)
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (€)")
	if sys.argv[1] == "data.csv":
		vehicle = "cars"
	else:
		vehicle = "cb500f motorbikes"
	plt.title(f"Scatter plot of {vehicle} prices according to their mileage")
	plt.savefig("Scatter_datas.png")
	plt.clf()


def plotting_lr(x_val, y_val, t0, t1):
	lr = t0 + t1 * x_val
	plt.scatter(x_val, y_val)
	plt.plot(x_val, lr, color='red')
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (€)")
	if sys.argv[1] == "data.csv":
		vehicle = "cars"
	else:
		vehicle = "cb500f motorbikes"
	plt.title(f"Scatter plot of {vehicle} prices according to their mileage, with the linear regression")
	plt.savefig("Scatter_with_lr.png")
	plt.clf()


def process_data(datas):
	datas = np.array(datas) # to pass from list of lists to array
	x = datas[:, 0]
	y = datas[:, 1]
	plotting_first(x, y)
	m = len(x) # m is the number of elements

	x_std = np.std(x)
	y_std = np.std(y)
	x_mean = np.mean(x)
	y_mean = np.mean(y)

	if x_std != 0 :
		x_stand = (x - x_mean)/x_std
		y_stand = (y - y_mean)/y_std
	else:
		raise ZeroDivisionError()
	return x, y, x_stand, y_stand, m


def main():
	try:
		assert len(sys.argv) == 2, "You must enter a correct dataset filename."
		datas = load_data(sys.argv[1])
		x, y, X, Y, m = process_data(datas)
		t0, t1, costs = gradient_descent_algo(X, Y, m, 0.01, 1000)
		t0, t1 = unstandardize(x, y, t0, t1)
		plotting_lr(x, y, t0, t1)
		print(f"theta0 = {t0}, theta1 = {t1}")
		print("The model has been trained and thetas have been saved in train_data.json")

		with open("train_data.json", "w") as file:
			json.dump({"theta0": t0, "theta1": t1}, file)

	except Exception as e:
		print(f"Error : {e}")
		return

if __name__ == "__main__":
	main()
