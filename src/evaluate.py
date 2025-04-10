import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import glob

# === Custom Dataset ===
class SkinDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        self.data['label'] = self.data[self.label_cols].values.argmax(axis=1)

        # Nur gültige Bildpfade und zugehörige Labels speichern
        image_paths = []
        labels = []

        for _, row in self.data.iterrows():
            image_id = row['image']
            label = row['label']

            pattern = os.path.join(self.image_dir, image_id + "*.jpg")
            matches = glob.glob(pattern)

            if matches:
                image_paths.append(matches[0])
                labels.append(label)

        self.image_paths = image_paths
        self.labels = labels
        print(f"[INFO] Loaded {len(self.image_paths)} valid image(s) from dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# === Modell laden ===
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# === Evaluation ===
def evaluate(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "resnet101_full_model.pth")
    csv_path = os.path.abspath(os.path.join(current_dir, "..", "data", "test.csv"))
    image_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "test_images"))

    model = load_model(model_path).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SkinDataset(csv_file=csv_path, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    acc = evaluate(model, dataloader, device)
    print(f"Accuracy on test set: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
