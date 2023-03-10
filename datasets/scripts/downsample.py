import os
import random
import shutil

# Define the original and new directories
original_dir = '../images'
new_dir = '../downsampled_images'

# Define the downsampling factor (1/10th in this case)
downsampling_factor = 10

# Create the new directory
os.makedirs(new_dir, exist_ok=True)

# Loop over each dataset
for dataset, size in [('HCI', 407), ('Concadia', 171), ('Pew', 9285), ('Statista', 34811)]:
    # Calculate the number of images to keep from this dataset
    n_keep = size // downsampling_factor

    # Get a list of all the filenames from this dataset
    filenames = [f for f in os.listdir(original_dir) if f.startswith(dataset.lower())]
    print(dataset, len(filenames))
    # Randomly select n_keep filenames from this dataset
    keep_filenames = random.sample(filenames, n_keep)

    # Copy the selected filenames to the new directory
    for filename in keep_filenames:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(new_dir, filename))
