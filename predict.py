import sys
import json
import os


def main():
    if len(sys.argv) != 2:
        print("Error: You must enter a mileage.")
        sys.exit()
    if not sys.argv[1].isdigit():
        print("Error: You must enter a mileage (positive whole number).")
        sys.exit()
    if os.path.exists("train_data.json"):
        with open("train_data.json", "r") as file:
            train_data = json.load(file)
            theta0 = train_data["theta0"]
            theta1 = train_data["theta1"]
    else:
        theta0 = 0
        theta1 = 0
    mileage = int(sys.argv[1])
    price = (theta0 * mileage) + theta1
    if price < 0:
        price = 0
    print(f"The estimated price for a car with {mileage} km is {price:.0f} €.")



if __name__ == '__main__':
    main()