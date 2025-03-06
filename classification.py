import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

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


"""
# Merge metadata with ground truth using "image" as the key
df = pd.merge(metadata, groundtruth, on="image")

print(df.columns)

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

print("Training data sample:")
print(train_df.head())

print("Testing data sample:")
print(test_df.head())"
"""