import sys


def main():
    if len(sys.argv) != 2:
        print("Error: You must enter a mileage.")
        sys.exit()
    if not sys.argv[1].isdigit():
        print("Error: You must enter a mileage (positive whole number).")
        sys.exit()
    try:



if __name__ == '__main__':
    main()