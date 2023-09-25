import os
import shutil

source_folder = 'Testing/embedding'
file_list = os.listdir(source_folder)

for filename in file_list:

    if os.path.isfile(os.path.join(source_folder, filename)):
        name, ext = os.path.splitext(filename)

        for i in range(2):
            new_filename = f"{name}__{i}{ext}"
            shutil.copy(os.path.join(source_folder, filename), os.path.join(source_folder, new_filename))
            print(f"Duplicate {i} created: {new_filename}")

print("Duplicating files completed.")
