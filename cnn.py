import os
from PIL import Image
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BORDER_PIXELS = 8 # number of border pixels processed by neural network per image fragment

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.3976, 0.4601, 0.5039), (0.2264, 0.2265, 0.2328)) # !!! PERHAPS PERFORM DYNAMIC CALCULATION?
])

# Dataset (one created for each image = 96 samples); for processing with DataLoader
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

# CNN
class CNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
      #self.bn1 = nn.BatchNorm2d(16)
      self.relu = nn.ReLU()
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
      #self.bn2 = nn.BatchNorm2d(32)
      self.fc1 = nn.Linear(32 * 32 * 4, 1)
      #self.dropout = nn.Dropout(p=0.2) # dropout of rate 0.35 for fully connected layer
      self.sigmoid = nn.Sigmoid()

   def forward(self, x): # fed a 128x16 image
      x = self.relu(self.conv1(x)) # 16 feature maps
      x = self.pool(x) # 16x64x8
      x = self.relu(self.conv2(x)) # 32 feature maps
      x = self.pool(x) # 32x32x4
      x = x.view(x.size(0), -1) # flatten
      x = self.sigmoid(self.fc1(x)) # 1 output
      return torch.squeeze(x) # return output as 0D tensor

# Takes image, returns dataset
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
            combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
         elif side == 3:
            if i >= 12: # no bottom matching piece
               continue
            combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
            combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
            combined_border.paste(quadrants[i + 4].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
            combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
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

   return PuzzleDataset(data, labels, transform) # return PyTorch dataset of image

# Given fragment index, retrieve a random non-bordering fragment
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

model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 32

train_path = "./dataset/train/" # path to train images
test_path = "./dataset/test/" # path to test images

try:
   train_images = os.listdir(train_path)
   test_images = os.listdir(test_path)

   # Iterate through train dataset
   #if not os.path.exists("cnn_model.pt"): # allows training to be done once
   if True:
      model.train()
      for index, file in enumerate(train_images):
         print(f"Training... ({index + 1}/{len(train_images)})") # display progress info
         
         dataset = create_dataset(Image.open(train_path + file)) # create dataset of image
         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

         # Train
         for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      torch.save(model, "cnn_model.pt") # save model (so doesn't have to be trained again, if desired)
   else:
      model = torch.load("cnn_model.pt") # load model already trained before
   all_preds, all_labels = [], [] # predictions and correct answers rsp.

   # Iterate through test dataset
   model.eval()
   for index, file in enumerate(test_images):
      print(f"Testing... ({index + 1}/{len(test_images)})")
      
      dataset = create_dataset(Image.open(test_path + file)) # create dataset from image
      loader = DataLoader(dataset, batch_size=batch_size)

      # Test
      with torch.no_grad():
         for images, labels in loader:
            outputs = torch.round(model(images)).int()
            all_preds.extend(outputs)
            all_labels.extend(labels)

   print(f"\nACCURACY: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
   print(f"PRECISION: {precision_score(all_labels, all_preds, pos_label=0) * 100:.2f}%")
   print(f"RECALL: {recall_score(all_labels, all_preds, pos_label=0) * 100:.2f}%")
   print(f"F1-SCORE: {f1_score(all_labels, all_preds, pos_label=0) * 100:.2f}%")

except Exception as e:
   print(f"An error occurred: {e}")

print("\nDone")