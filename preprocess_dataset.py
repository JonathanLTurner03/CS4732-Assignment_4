import os
import shutil

# Path directories
base_dir = 'DS_IDRID'
subfolders = ['Test', 'Train']

# Sets the labels to each category.
dr_labels = {'3', '4'}
nondr_labels = {'0'}


# Creates a subfolder for each one of the labels in each section
for subfolder in subfolders:
    dr_path = os.path.join(base_dir, subfolder, 'dr')
    nondr_path = os.path.join(base_dir, subfolder, 'nondr')
    os.makedirs(dr_path, exist_ok=True)
    os.makedirs(nondr_path, exist_ok=True)


# Filters these files into the sub directories and categories.
def move_files(subfolder):
    folder_path = os.path.join(base_dir, subfolder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            label = filename.split('-')[-1].split('.')[0]
            if label in dr_labels:
                shutil.move(os.path.join(folder_path, filename), os.path.join(folder_path, 'dr', filename))
            elif label in nondr_labels:
                shutil.move(os.path.join(folder_path, filename), os.path.join(folder_path, 'nondr', filename))
            else:
                os.remove(os.path.join(folder_path, filename))


move_files(subfolders[0])
move_files(subfolders[1])