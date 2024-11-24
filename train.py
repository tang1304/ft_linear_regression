import csv
import numpy as np

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

def process_data(datas):
	datas = np.array(datas) # to pass from list of lists to array
	x = datas[:, 0]
	y = datas[:, 1]
	m = len(x) # m is the number of elements
	

def main():
	try:
		datas = load_data('data.csv')
		process_data(datas)
	except ValueError as e:
		print(f"Error : ", e)
		return
	
if __name__ == "__main__":
	main()