import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

# Define paths
image_folder = "archive/ISIC_2019_Training_Input"
metadata_csv = "ISIC_2019_Training_GroundTruth.csv"
groundtruth_csv = "ISIC_2019_Training_Metadata.csv"

# Load CSV files
metadata_df = pd.read_csv(metadata_csv)
groundtruth_df = pd.read_csv(groundtruth_csv)

# Merge data on image filename
df = metadata_df.merge(groundtruth_df, on="filename")

# Get list of all image paths
image_paths = {os.path.basename(p): p for p in glob.glob(os.path.join(image_folder, "*"))}

# Add full image paths to dataframe
df["image_path"] = df["filename"].map(image_paths)

# Drop rows with missing images
df = df.dropna(subset=["image_path"])

# Split into Train/Test (70/30)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["groundtruth_column"]) 

# Print results
print(f"Total: {len(df)}, Training: {len(train_df)}, Testing: {len(test_df)}")

# Save train and test splits (optional)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
