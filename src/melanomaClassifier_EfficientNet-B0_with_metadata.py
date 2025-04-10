# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Eigene Dataset-Klasse für PyTorch
class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

        # Extrahiere und skaliere die Metadaten
        meta_features = dataframe[['age_approx', 'sex_encoded', 'anatom_site_general_challenge_encoded']].values
        self.meta = torch.tensor(meta_features, dtype=torch.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['image'])
        image = Image.open(img_name).convert("RGB")
        label = int(self.dataframe.iloc[idx]['label'])  # Label als Integer
        meta = self.meta[idx]  # Metadaten

        if self.transform:
            image = self.transform(image)

        return image, meta, label


# Kombiniertes Modell aus EfficientNet und Metadaten
class MetaEfficientNet(nn.Module):
    def __init__(self, meta_input_dim):
        super(MetaEfficientNet, self).__init__()
        from torchvision.models import efficientnet_b0
        self.base_model = efficientnet_b0(pretrained=True)

        # Blöcke 0–5 bleiben eingefroren, Blöcke 6–7 werden trainiert
        for i, block in enumerate(self.base_model.features):
            if i >= 5:
                for param in block.parameters():
                    param.requires_grad = True
            else:
                for param in block.parameters():
                    param.requires_grad = False

        self.base_model.classifier[1] = nn.Identity()  # entferne finale Klassifizierung

        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1280 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8)
        )

    def forward(self, image, meta):
        image_features = self.base_model(image)
        meta_features = self.meta_net(meta)
        combined = torch.cat((image_features, meta_features), dim=1)
        return self.classifier(combined)


def main():
    ########################################
    # Load and normalize train and Test data
    ########################################

    # Datenpfade
    IMAGE_DIR = "E:\\melanoma_model\\ISIC_2019_Training_Input"
    LABELS_CSV = "C:\\Users\\gabri\\OneDrive\\Studium\\kurse\\6_Semester\\dshealth\\MelanomaClassification\\ISIC_2019_Training_GroundTruth.csv"
    META_CSV = "C:\\Users\\gabri\\OneDrive\\Studium\\kurse\\6_Semester\\dshealth\\MelanomaClassification\\ISIC_2019_Training_Metadata.csv"

    # CSV-Dateien einlesen
    df = pd.read_csv(LABELS_CSV)
    df_meta = pd.read_csv(META_CSV)

    df['image'] = df['image'] + ".jpg"  # Bildnamen anpassen (falls nötig)
    df = pd.merge(df, df_meta, on='image')
    print(df.head())  # Ersten paar Zeilen anzeigen

    # Liste der Label-Spalten in der richtigen Reihenfolge
    label_cols = [
        'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'
    ]

    # Neue Spalte mit der Klassen-ID (0 bis 8)
    df['label'] = df[label_cols].values.argmax(axis=1)

    # Metadaten vorbereiten
    df['sex_encoded'] = LabelEncoder().fit_transform(df['sex'].fillna('unknown'))
    df['anatom_site_general_challenge_encoded'] = LabelEncoder().fit_transform(df['anatom_site_general_challenge'].fillna('unknown'))
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    df['age_approx'] = StandardScaler().fit_transform(df[['age_approx']])

    # Transformationen für Bilder (VGG16 erwartet bestimmte Normalisierung)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset laden
    dataset = SkinCancerDataset(df, IMAGE_DIR, transform=transform)

    # Split in Train (70%), Validation (15%), Test (15%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Rest

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # für Reproduzierbarkeit
    )

    # DataLoader für Batch-Verarbeitung
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)}, Validierung: {len(val_dataset)}")

    ###############################################
    # Define the convolutional Neural Network (CNN)
    ###############################################

    # Prüfen, ob CUDA (GPU) verfügbar ist
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Prints "cuda" if GPU is available, otherwise "cpu"

    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(pretrained=True)

    # Blöcke 0–5 bleiben eingefroren, Blöcke 6–7 werden trainiert
    for i, block in enumerate(model.features):
        if i >= 5:
            for param in block.parameters():
                param.requires_grad = True
        else:
            for param in block.parameters():
                param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)

    model = model.to(device)  # Modell auf GPU/CPU verschieben

    ###############################################
    # Define the loss funtion and optimizer
    ###############################################

    weights = torch.tensor([0.01864082, 0.00654709, 0.02536677, 0.09722467,
                            0.03212416, 0.35269368, 0.33317703, 0.13422578]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.00008)  # Learning Rate 0.0001

    ###############################################
    # Train the model on train data
    ###############################################

    # Check device of the model
    print("Model is on:", next(model.parameters()).device)
    inputs, meta, labels = next(iter(train_loader))
    inputs = inputs.to(device)
    meta = meta.to(device)
    labels = labels.to(device)
    print("Batch is on:", inputs.device)

    epochs = 75
    patience = 7
    best_acc = 0.0
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            print("\u23f9\ufe0f Frühes Stoppen ausgelöst – Training beendet.")
            break

        model.train()
        running_loss = 0.0

        for images, meta, labels in train_loader:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validierung
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, meta, labels in val_loader:
                images, meta, labels = images.to(device), meta.to(device), labels.to(device)

                outputs = model(images, meta)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Validation Acc: {acc:.4f}")

        # 🔍 Early Stopping prüfen
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"\ud83d\udd52 Keine Verbesserung seit {epochs_no_improve} Epochen")

            if epochs_no_improve >= patience:
                early_stop = True

    ###############################################
    # evaluate the model on the test data
    ###############################################

    # Modell in den Evaluationsmodus setzen
    # evaluate the model on the test data
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, meta, labels in test_loader:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            outputs = model(images, meta)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # In NumPy-Arrays umwandeln
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metriken berechnen
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(classification_report(all_labels, all_preds, target_names=label_cols))

    # Ergebnisse ausgeben
    print(f"\u2705 Accuracy: {accuracy:.4f}")
    print(f"\u2705 Precision: {precision:.4f}")
    print(f"\u2705 Recall: {recall:.4f}")
    print(f"\u2705 F1-Score: {f1:.4f}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # optional, good practice on Windows
    main()
