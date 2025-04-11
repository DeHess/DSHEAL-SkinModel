# DSHEAL-SkinModel

This project explores the use of deep convolutional neural networks (CNNs) for classifying dermoscopic images of skin lesions, with a focus on melanoma detection. Leveraging transfer learning, we evaluated several pre-trained CNN architectures and ultimately selected ResNet-101 due to its superior initial performance. The model was trained and fine-tuned on the ISIC 2019 dataset, which presents challenges such as class imbalance and subtle inter-class variation. Our work highlights the effectiveness of deep learning for multi-class medical image classification and the importance of robust evaluation metrics in imbalanced datasets.

# Dataset Details
Source: ISIC 2019 - Skin Lesion Images for Melanoma Classification (Kaggle)

Size: 25,331 labeled dermoscopic images

Classes: 9 total — Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion, Squamous cell carcinoma, None of the above

Preprocessing:

Removed “UNK” class

Images resized to 224×224 px

Normalized to match pre-trained model input

Label encoding used (no one-hot)

No augmentation applied (found ineffective in tests)

# Installation and Usage Instructions
Clone the repository and ensure all dataset paths are correctly set to your local environment.

Install dependencies (e.g., PyTorch, NumPy, scikit-learn, matplotlib).

Run the Jupyter Notebook or Python script to train and evaluate the model.

# Model Performance Summary
Best Model: ResNet-101 (fine-tuned)

Test Accuracy: 79.8%

Weighted Avg F1-Score: 0.80

Macro Avg F1-Score: 0.68

Strong Classes: Melanocytic nevus (F1 = 0.89), Basal cell carcinoma (F1 = 0.82)

Challenging Classes: Actinic keratosis (F1 = 0.37), Squamous cell carcinoma (F1 = 0.43)

ROC AUC: Most classes > 0.95; best AUC = 0.98 for class 2 and class 6

Notes: Class weighting helped poor-performing classes slightly but lowered overall accuracy


# Link To OneDrive

https://1drv.ms/f/s!AtRU6kKnPW-cqa9lMcpjIjs6coM5kA?e=oCOh5c

# Team Members and Contributions
Gabriel Gillmann
Nathan Hess


