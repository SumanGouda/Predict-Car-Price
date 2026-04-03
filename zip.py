import shutil
import os

# 1. The name of your existing folder containing all city folders
folder_to_zip = r"C:\Users\LENOVO\Downloads\My_Car_Dataset"

# 2. The name you want for your final zip file (e.g., car_data_export.zip)
output_filename = 'kaggle_car_dataset'

# This one line creates the zip file
shutil.make_archive(output_filename, 'zip', folder_to_zip)

print(f"Success! {output_filename}.zip has been created in your current directory.")