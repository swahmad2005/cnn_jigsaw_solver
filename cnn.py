import os
from PIL import Image
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score

BORDER_PIXELS = 8 # number of border pixels processed by neural network per image fragment

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4152, 0.4650, 0.5014), (0.2298, 0.2304, 0.2357))
])

class CNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1)
      self.relu = nn.ReLU()
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
      self.fc1 = nn.Linear(32 * 32 * 4, 1)
      self.sigmoid = nn.Sigmoid()

   def forward(self, x): # fed a 128x16 image
      x = self.relu(self.conv1(x)) # 16 feature maps
      x = self.pool(x) # 16x64x8
      x = self.relu(self.conv2(x)) # 32 feature maps
      x = self.pool(x) # 32x32x4
      x = x.view(-1) # flatten
      x = self.sigmoid(self.fc1(x)) # 1 output
      return x

model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_path = "./dataset/train/" # path to train images
test_path = "./dataset/test/" # path to test images

try:
   train_images = os.listdir(train_path)
   test_images = os.listdir(test_path)

   # Iterate through train dataset
   for index, file in enumerate(train_images):
      print(f"Training... ({index + 1}/{len(train_images)})")
      train_img = Image.open(train_path + file)
      #train_img = train_img.convert("RGB")
      
      # Split image into 16 quadrants
      quadrants = []
      for row in range(4):
         for col in range(4):
            quadrants.append(train_img.crop((col * 128, row * 128, col * 128 + 128, row * 128 + 128)))

      training_data = [] # consists of 96 samples per image with correct output (48 matching, 48 non-matching); tuple of (sample, output); output=0 means match, output=1 means non-match

      # Gather all 48 matching borders
      total=0
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
            elif side == 3:
               if i >= 12: # no bottom matching piece
                  continue
               combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
               combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
               combined_border.paste(quadrants[i + 4].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
            #combined_border.save(f"./training/match_{index}_{total}.jpg")
            training_data.append((combined_border, 0))
            total+=1

      # Gather 48 random non-matching borders
      total=0
      for _ in range(48):
         i = random.randint(0, 15) # select random fragment
         j = 15 - i # simple way to retrieve non-bordering fragment (works specifically for 16 quadrants)
         side = random.randint(0, 3)

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
         elif side == 3:
            combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
            combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
            combined_border.paste(quadrants[j].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
         #combined_border.save(f"./training/fail_{index}_{total}.jpg")
         training_data.append((combined_border, 1))
         total+=1

      # Train
      random.shuffle(training_data)
      for img, label in training_data:
         tensor = transform(img)
         output = model(tensor)
         loss = criterion(output, torch.tensor([float(label)]))
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

   total_guesses = 0 # for testing accuracy
   total_correct = 0 # for testing accuracy

   # Iterate through test dataset
   for index, file in enumerate(test_images):
      print(f"Testing... ({index + 1}/{len(test_images)})")
      test_img = Image.open(test_path + file)

      # Split image into 16 quadrants
      quadrants = []
      for row in range(4):
         for col in range(4):
            quadrants.append(test_img.crop((col * 128, row * 128, col * 128 + 128, row * 128 + 128)))

      testing_data = [] # consists of 96 samples per image with correct output (48 matching, 48 non-matching); tuple of (sample, output); output=0 means match, output=1 means non-match

      # Gather all 48 matching borders
      total=0
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
            elif side == 3:
               if i >= 12: # no bottom matching piece
                  continue
               combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
               combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
               combined_border.paste(quadrants[i + 4].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
            #combined_border.save(f"./training/match_{index}_{total}.jpg")
            testing_data.append((combined_border, 0))
            total+=1
      
      # Gather 48 random non-matching borders
      total=0
      for _ in range(48):
         i = random.randint(0, 15) # select random fragment
         j = 15 - i # simple way to retrieve non-bordering fragment (works specifically for 16 quadrants)
         side = random.randint(0, 3)

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
         elif side == 3:
            combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
            combined_border.paste(quadrants[i].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
            combined_border.paste(quadrants[j].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
         #combined_border.save(f"./training/fail_{index}_{total}.jpg")
         testing_data.append((combined_border, 1))
         total+=1

      # Test
      for img, label in testing_data:
         tensor = transform(img)
         output = round(model(tensor).item())
         total_guesses += 1
         if label == output:
            total_correct += 1
   
   print(f"TOTAL ACCURACY: {(100 * total_correct / total_guesses):.2f}% ({total_correct}/{total_guesses})")

except Exception as e:
   print(f"An error occurred: {e}")

print("Done")

# STATS
# 0 training images, 10 testing images (5 runs): 50.62%, 50.21%, 42.92%, 43.23%, 49.69%
# 1 training image, 10 testing images (5 runs): 63.54%, 59.06%, 57.81%, 55.00%, 61.15%
# 10 training image, 10 testing images (5 runs): 83.65%, 83.65%, 85.10%, 85.00%, 81.67%
# 50 training image, 10 testing images (5 runs): 84.90%, 89.39%, 87.92%, 91.56%, 91.46%

# 50 training image, 1500 testing images (3 runs): 89.61%, 90.86%, 91.17%
# 100 training image, 1500 testing images (5 runs): 94.44%, 96.04%, 94.56%, 95.28%, 96.22%
# 300 training image, 1500 testing images (5 runs): 94.72%, 93.55%, 94.29%, 94.60%, 93.69%

# 8 border pixel: 97.30%, 96.81%, 96.63%, 97.50%, 95.25%
# another FC: 97.14%, 96.19%, 94.42%
# 2 poolings (instead of 3): 97.79%