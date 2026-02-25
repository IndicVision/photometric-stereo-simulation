import pandas as pd

CSV_FILE = "measurements.csv"

def add_row(sample_num, length, breadth):
    df = pd.read_csv(CSV_FILE)

    area = length * breadth

    new_row = {
        "Sample Number": sample_num,
        "Length": length,
        "Breadth": breadth,
        "Area": area
    }

    df.loc[len(df)] = new_row
    df.to_csv(CSV_FILE, index=False)

    return area

def main():
    while True:
        sample_num = input("Sample number (or 'q' to quit): ")
        if sample_num.lower() == "q":
            break

        length = float(input("Length: "))
        breadth = float(input("Breadth: "))

        area = add_row(sample_num, length, breadth)
        print(f"Area: {area} - Entry saved!\n")

    print("Data saved to measurements.csv")

if __name__ == "__main__":
    main()
