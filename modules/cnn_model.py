# 'cnn_model.py' contains the class definition of the CNN itself, transforms, training and testing

import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BORDER_PIXELS = 8 # number of border pixels processed by neural network per image fragment (this yielded the best results)
unscrambling_image = False

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3976, 0.4601, 0.5039), (0.2265, 0.2265, 0.2329)) # predetermined values from all training images
])

# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 30 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # feed a 128x16 image
        x = self.relu(self.conv1(x)) # 16 feature maps
        x = self.pool(x)
        x = self.relu(self.conv2(x)) # 32 feature maps
        x = self.pool(x)
        if not unscrambling_image:
            x = x.view(x.size(0), -1) # flatten with batches (default)
        else:
            x = torch.flatten(x) # flatten without batches (when used by 'unscramble_images.py')
        x = self.sigmoid(self.fc1(x)) # 1 output
        return torch.squeeze(x) # return output as 0D tensor
        
# Dataset (one created per each image = 96 samples); processed with DataLoader
class PuzzleDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.transform(self.data[idx])
        label = self.labels[idx]
        return sample, label

# Creates a usable dataset (for both training/testing) per image
def create_dataset(img):
    # Split image into 16 quadrants
    quadrants = [] # contains all 16 quadrants
    for row in range(4):
        for col in range(4):
            quadrants.append(img.crop((col * 128, row * 128, col * 128 + 128, row * 128 + 128))) # each quadrant is 128x128

    data = [] # consists of 96 samples per image with correct output (48 matching, 48 non-matching)
    labels = [] # output=0 means match, output=1 means non-match

    # Gather all 48 matching borders
    for i in range(16):
        for side in range(4):
            combined_border = None
            if side == 0:
                if i % 4 == 0: # no left matching piece
                    continue
                combined_border = Image.new("RGB", (2 * BORDER_PIXELS, 128))
                combined_border.paste(quadrants[i - 1].crop((128 - BORDER_PIXELS, 0, 128, 128)), (0, 0))
                combined_border.paste(quadrants[i].crop((0, 0, BORDER_PIXELS, 128)), (BORDER_PIXELS, 0))
            elif side == 1:
                if i % 4 == 3: # no right matching piece
                    continue
                combined_border = Image.new("RGB", (2 * BORDER_PIXELS, 128))
                combined_border.paste(quadrants[i].crop((128 - BORDER_PIXELS, 0, 128, 128)), (0, 0))
                combined_border.paste(quadrants[i + 1].crop((0, 0, BORDER_PIXELS, 128)), (BORDER_PIXELS, 0))
            elif side == 2:
                if i <= 3: # no top matching piece
                    continue
                combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
                combined_border.paste(quadrants[i - 4].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
                combined_border.paste(quadrants[i].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
                combined_border = combined_border.transpose(Image.TRANSPOSE) # ensure 128x16
            elif side == 3:
                if i >= 12: # no bottom matching piece
                    continue
                combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
                combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
                combined_border.paste(quadrants[i + 4].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
                combined_border = combined_border.transpose(Image.TRANSPOSE) # ensure 128x16
            data.append(combined_border)
            labels.append(0)

    # Gather 48 random non-matching borders
    for _ in range(48):
        i = random.randint(0, 15) # select random fragment
        j = random_nonmatch(i) # select random non-bordering fragment
        side = random.randint(0, 3) # select random side

        combined_border = None
        if side == 0:
            combined_border = Image.new("RGB", (2 * BORDER_PIXELS, 128))
            combined_border.paste(quadrants[i].crop((128 - BORDER_PIXELS, 0, 128, 128)), (0, 0))
            combined_border.paste(quadrants[j].crop((0, 0, BORDER_PIXELS, 128)), (BORDER_PIXELS, 0))
        elif side == 1:
            combined_border = Image.new("RGB", (2 * BORDER_PIXELS, 128))
            combined_border.paste(quadrants[i].crop((128 - BORDER_PIXELS, 0, 128, 128)), (0, 0))
            combined_border.paste(quadrants[j].crop((0, 0, BORDER_PIXELS, 128)), (BORDER_PIXELS, 0))
        elif side == 2:
            combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
            combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
            combined_border.paste(quadrants[j].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
            combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
        elif side == 3:
            combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
            combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
            combined_border.paste(quadrants[j].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
            combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
        data.append(combined_border)
        labels.append(1)

    return PuzzleDataset(data, labels, transform) # return valid PyTorch dataset

# Given a fragment index, retrieve a random non-bordering fragment
def random_nonmatch(idx): 
    exclude = {idx} # find all indices to exclude (start with self)
    if idx % 4 != 0: # exclude left
        exclude.add(idx - 1)
    if idx % 4 != 3: # exclude right
        exclude.add(idx + 1)
    if idx >= 4: # exclude top
        exclude.add(idx - 4)
    if idx <= 11: # exclude bottom
        exclude.add(idx + 4)
    return random.choice(list(set(range(15)) - exclude)) # return any index except excluded ones

# Train CNN model
def train_model():
    try:
        model = CNN()
        model.train()
        criterion = nn.MSELoss() # Mean Squared Error loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam optimizer

        train_path = "./dataset/train/"
        train_images = os.listdir(train_path)

        for index, file in enumerate(train_images):
            print(f"Training... ({index + 1}/{len(train_images)})")
            
            with Image.open(train_path + file) as img:
                dataset = create_dataset(img) # create dataset of image
                loader = DataLoader(dataset, batch_size=96, shuffle=True) # batch contains all samples per image

                # Train
                for images, labels in loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        torch.save(model, "cnn_trained.pt") # save model to file
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Test CNN model
def test_model(model_name):
    try:
        model = CNN()
        model = torch.load(model_name)
        model.eval()

        test_path = "./dataset/test/"
        test_images = os.listdir(test_path)

        all_preds, all_labels = [], [] # predictions and correct answers rsp.
        for index, file in enumerate(test_images):
            print(f"Testing... ({index + 1}/{len(test_images)})")
            
            with Image.open(test_path + file) as img:
                dataset = create_dataset(img) # create dataset of image
                loader = DataLoader(dataset, batch_size=96) # batch contains all samples per image

                # Test
                with torch.no_grad():
                    for images, labels in loader:
                        outputs = torch.round(model(images)).int() # prediction is rounded to 0 or 1
                        all_preds.extend(outputs)
                        all_labels.extend(labels)

        # Show how well the model did
        print(f"\nACCURACY: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
        print(f"PRECISION: {precision_score(all_labels, all_preds, pos_label=0) * 100:.2f}%")
        print(f"RECALL: {recall_score(all_labels, all_preds, pos_label=0) * 100:.2f}%")
        print(f"F1-SCORE: {f1_score(all_labels, all_preds, pos_label=0) * 100:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {e}")