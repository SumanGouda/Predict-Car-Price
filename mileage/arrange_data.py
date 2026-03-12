import json
import os

req_col = [
    'Mileage',
    'Engine',
    'Kerb Weight',
    'Fuel',
    'Transmission Type',
    'Power',
    'No. of Cylinders',
    'Registration Year'
]

def arrange_data(json_path):

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Folder where you want the CSV
    folder = r"D:\IMP  ML  PROJECTS\CAR PRICE PREDICTION\mileage"

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Full path of CSV
    csv_path = os.path.join(folder, "dataset.csv")

    # Write CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(req_col) + "\n")

        for car in data:
            row = [str(car.get(col, "")) for col in req_col]
            f.write(",".join(row) + "\n")

    print(f"CSV saved at: {csv_path}")


if __name__ == "__main__":
    json_file = r"D:\IMP  ML  PROJECTS\CAR PRICE PREDICTION\web scraping\extraction\car_dataset_ahmedabad.json"
    arrange_data(json_file)