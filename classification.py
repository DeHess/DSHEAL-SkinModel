import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(1022, 767))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

def preprocess_metadata(df):
    label_encoders = {}
    categorical_columns = ['sex', 'anatom_site_general', 'lesion_id']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    df['age_approx'] = scaler.fit_transform(df[['age_approx']])

    return df


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

# Split into Train and Test Dataset
df = pd.merge(metadata, groundtruth, on="image")
train_df, test_df = train_test_split(df, test_size=0.3)

train_df = preprocess_metadata(train_df)
test_df = preprocess_metadata(test_df)


