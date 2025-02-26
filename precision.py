from train import load_data
import numpy as np
import json
import os
import sys


def predict(t0, t1, mileages):
	return [int((t1 * mileage) + t0) for mileage in mileages]


def calculate_mse(real_prices, predicted_prices, m) -> float:
	errors = real_prices - predicted_prices
	squared_errors = errors ** 2
	return (1 / m) * np.sum(squared_errors)


def main():
	try:
		assert len(sys.argv) == 2, "You must enter a correct dataset filename."
		datas = load_data(sys.argv[1])
		assert os.path.exists("train_data.json"), "Error: train_data.json not found"
		with open("train_data.json", "r") as file:
			train_data = json.load(file)
			theta0 = train_data["theta0"]
			theta1 = train_data["theta1"]
		datas = np.array(datas)
		mileages = datas[:, 0]
		real_prices = datas[:, 1]
		predicted_prices = predict(theta0, theta1, mileages)
		m = len(mileages)

		mse = calculate_mse(real_prices, predicted_prices, m)
		print(f"Mean Squared Error: {mse:.2f}")
		print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
		print(f"So the average error of our prediction compared to the real prices is {np.sqrt(mse):.2f} €")

	except Exception as e:
		print(f"Error : {e}")
		return


if __name__ == "__main__":
	main()
