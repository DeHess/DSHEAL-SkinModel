import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

image_folder = "archive/ISIC_2019_Training_Input"
metadata_csv = "archive/ISIC_2019_Training_Metadata.csv"
groundtruth_csv = "archive/ISIC_2019_Training_GroundTruth.csv"

metadata = pd.read_csv(metadata_csv)
groundtruth = pd.read_csv(groundtruth_csv)

# Drop the 'UNK' class column from groundtruth. 
# UNK stands for UNKNOWN, or "none of the above" and can
# be assigned last, if no other diagnosis was found.
groundtruth = groundtruth.drop(columns=["UNK"])

# Fill with "Unknown" if feature is empty
metadata["anatom_site_general"] = metadata["anatom_site_general"].fillna("unknown")
metadata["lesion_id"] = metadata["lesion_id"].fillna("unknown")
metadata["sex"] = metadata["sex"].fillna("unknown")

# Use the median age if no age is specified 
metadata["age_approx"] = metadata["age_approx"].fillna(metadata["age_approx"].median())

# Merge metadata with ground truth using "image" as the key
df = pd.merge(metadata, groundtruth, on="image")

# Split into train (70%) and test (30%)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df.iloc[:, 1:]) 

# Save the train and test datasets
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")