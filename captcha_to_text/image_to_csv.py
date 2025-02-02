
import os
import pandas as pd

# Path to your folder of test images
test_folder = "/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Datasets/test_folder_v3"

# CSV output path
csv_output_path = "/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Datasets/test2.csv"

# Create a list to store image paths and labels
data = []

# Iterate through the images in the folder
for filename in os.listdir(test_folder):
    if filename.endswith(".png"):
        # Extract the label from the filename (e.g., assuming format "label.png")
        label = os.path.splitext(filename)[0]  # Remove file extension
        # Full path to the image
        image_path = os.path.join(test_folder, filename)
        # Append to the list
        data.append([image_path, label])

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=["image_path", "label"])

# Save the DataFrame as a CSV file
df.to_csv(csv_output_path, index=False)

print(f"CSV file saved at {csv_output_path}")
