import csv

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
	print(datas)

def main():
	try:
		load_data('data.csv')
	except ValueError as e:
		print(f"Error : ", e)
		return
	
if __name__ == "__main__":
	main()