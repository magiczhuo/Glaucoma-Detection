import kagglehub
import os

# Define the dataset handle and base name for the directory
handle = "deathtrooper/multichannel-glaucoma-benchmark-dataset"
base_name = "SMDG"

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Define the initial valid directory path
target_dir = os.path.join(project_root, base_name)

# Ensure unique directory name to avoid conflicts
counter = 1
while os.path.exists(target_dir):
    target_dir = os.path.join(project_root, f"{base_name}_{counter}")
    counter += 1

print(f"Downloading dataset to: {target_dir}")

# Ensure the directory exists
os.makedirs(target_dir, exist_ok=True)

try:
    path = kagglehub.dataset_download(
        handle,
        path=None,
        force_download=True,
        output_dir=target_dir
    )

    print("Dataset downloaded to:", path)
    print("Files in dataset:")
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))

except Exception as e:
    print(f"An error occurred: {e}")
    # Optional: cleanup empty directory if failed
    if os.path.exists(target_dir) and not os.listdir(target_dir):
        os.rmdir(target_dir)

        print(os.path.join(root, file))
