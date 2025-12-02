# Finds the total differences in all RGB values for pixels along the borders for all image fragments
# Border with the minimum RGB difference is predicted to be the matching image fragment
#
# Does not require training phase (because this is not a neural network, this is a simple algorithm)
# SERVES AS A BENCHMARK; goal is to use neural networks to enhance border loss calculations to get better results
#
# Note: image size always 512x512, image fragment size always 128x128 (16 fragments per image)
# 960 possible comparisons per image (but 480 unique since each is duplicated)

import os
from PIL import Image
import random
from collections import Counter
import torch
import torch.nn as nn
from torchvision import transforms


BORDER_PIXELS = 8 # number of border pixels processed by neural network per image fragment

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3976, 0.4601, 0.5039), (0.2264, 0.2265, 0.2328)) # !!! PERHAPS PERFORM DYNAMIC CALCULATION?
])

# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 32 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # fed a 128x16 image
        x = self.relu(self.conv1(x)) # 16 feature maps
        x = self.pool(x) # 16x64x8
        x = self.relu(self.conv2(x)) # 32 feature maps
        x = self.pool(x) # 32x32x4
        #x = x.view(x.size(0), -1) # flatten
        x = torch.flatten(x) # flatten
        x = self.sigmoid(self.fc1(x)) # 1 output
        return torch.squeeze(x) # return output as 0D tensor
    
# Two borders as inputs (128x3), computes total RGB difference (smaller value = more resemblance)
def border_loss(b1, b2):
    diff = 0
    for i in range(127):
        #diff += abs(b1[i][0] - b2[i][0]) + abs(b1[i][1] - b2[i][1]) + abs(b1[i][2] - b2[i][2])
        diff += ((b1[i][0] - b2[i][0]) ** 2 + (b1[i][1] - b2[i][1]) ** 2 + (b1[i][2] - b2[i][2]) ** 2) ** 0.5
    return diff

# Locates the minimum loss in the predicted borders (identifies the next fragment to be replaced into the puzzle, as this is what the algorithm is the most confident in)
def min_predicted_loss(list):
    min_info = (None, None, float('inf')) # tracks minimum loss as tuple: (fragment, side, loss)
    for fragment in range(16):
        for side in range(4):
            if (list[fragment][side][0] < min_info[2]): # new min_loss found
                min_info = (fragment, side, list[fragment][side][0])
    list[fragment][side] = (float('inf'), None) # remove the predicted piece as it now has been selected
    return min_info

# Represents a puzzle fragment and its connections (used to determine whether algorithm successfully reconstructed image)
class Fragment:
    def __init__(self, id, left, right, top, bottom):
        self.id = id
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def show_neighbors(self):
        return f"{self.id} - (Left: {self.left}; Right: {self.right}; Top: {self.top}; Bottom: {self.bottom})"
    
    def __str__(self):
        return str(self.id)

# Distributes border losses from a scale of 0 to 1 relatively (lower = more confident; losses that are much lower than other relative losses get lower overall value)
def confidence_dist(list):
    sum = 0
    for n in list: # find sum (exclude infinity)
        if n != float('inf'):
            sum += n
    for i in range(len(list)):
        list[i] /= sum
    return list

path = "./dataset/test/" # path to test images
path = "./sample2/"

accuracies = [] # tracks border comparison accuracies
images_recreated = 0 # images successfully reconstructed

model = CNN()

try:
    entries = os.listdir(path) # retrieve images

    model = torch.load("99.66.pt") # load model already trained before

    # Perform tests on each image
    for index, file in enumerate(entries):
        print(f"Testing image ({index + 1}/{len(entries)})")
        img = Image.open(path + file) # open image
        img = img.resize((512, 512)) # resize in case not 512x512

        # Split image into 16 quadrants
        quadrants = [] # contains all 16 quadrants
        for row in range(4):
            for col in range(4):
                quadrants.append(img.crop((col * 128, row * 128, col * 128 + 128, row * 128 + 128))) # each quadrant is 128x128
        
        # Scramble image
        random.seed(index) # custom seed (so can check later if correct)
        random.shuffle(quadrants)
        
        # Create CNN inputs (all 960 possible border connections)
        outputs = [] # stores all inputs to be fed into CNN
        for f1 in range(16): # iterate through each fragment
            fragment_border = []
            for side in range(4): # iterate through all 4 sides of fragment
                side_border = []
                for f2 in range(16): # compare to all other fragments
                    if f1 >= f2: # omit comparing fragment to self, or recomparing fragments
                        outputs.append(float('inf'))
                        continue
                    elif side == 0:
                        combined_border = Image.new("RGB", (2 * BORDER_PIXELS, 128))
                        combined_border.paste(quadrants[f2].crop((128 - BORDER_PIXELS, 0, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f1].crop((0, 0, BORDER_PIXELS, 128)), (BORDER_PIXELS, 0))
                    elif side == 1:
                        combined_border = Image.new("RGB", (2 * BORDER_PIXELS, 128))
                        combined_border.paste(quadrants[f1].crop((128 - BORDER_PIXELS, 0, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f2].crop((0, 0, BORDER_PIXELS, 128)), (BORDER_PIXELS, 0))
                    elif side == 2:
                        combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
                        combined_border.paste(quadrants[f2].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f1].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
                        combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
                    elif side == 3:
                        combined_border = Image.new("RGB", (128, 2 * BORDER_PIXELS))
                        combined_border.paste(quadrants[f1].crop((0, 128 - BORDER_PIXELS, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f2].crop((0, 0, 128, BORDER_PIXELS)), (0, BORDER_PIXELS))
                        combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
                    outputs.append(model(transform(combined_border)).item())

        # Attach most resembling fragments (top 24)
        reconstructed_fragments = [Fragment(i, None, None, None, None) for i in range(16)]
        #min_fragment, min_side, min_loss = None, None, float('inf') # track minimum fragment/side based on minimum loss
        for _ in range(24):
            min_loss = min(outputs) # most resembling border
            min_index = outputs.index(min_loss)
            outputs[min_index] = float('inf') # find next most resembling border
            f1, side, f2 = min_index // 64, min_index % 64 // 16, min_index % 16

            # Place the predicted fragment in the corresponding location (based on fragment/side of minimum loss)
            if side == 0: # best comparison found for a left border
                if reconstructed_fragments[f1].left == None and reconstructed_fragments[f2].right == None:
                    reconstructed_fragments[f1].left = reconstructed_fragments[f2]
                    reconstructed_fragments[f2].right = reconstructed_fragments[f1]
            elif side == 1: # best comparison found for a right border
                if reconstructed_fragments[f1].right == None and reconstructed_fragments[f2].left == None:
                    reconstructed_fragments[f1].right = reconstructed_fragments[f2]
                    reconstructed_fragments[f2].left = reconstructed_fragments[f1]
            elif side == 2: # best comparison found for a top border
                if reconstructed_fragments[f1].top == None and reconstructed_fragments[f2].bottom == None:
                    reconstructed_fragments[f1].top = reconstructed_fragments[f2]
                    reconstructed_fragments[f2].bottom = reconstructed_fragments[f1]
            elif side == 3: # best comparison found for a bottom border
                if reconstructed_fragments[f1].bottom == None and reconstructed_fragments[f2].top == None:
                    reconstructed_fragments[f1].bottom = reconstructed_fragments[f2]
                    reconstructed_fragments[f2].top = reconstructed_fragments[f1]

        # Find top-left fragment
        top_left_fragment = reconstructed_fragments[0] # random starting fragment
        for _ in range(6): # retrieve top-left most fragment (must take 6 moves or fewer if all fragments were successfully connected)
            if top_left_fragment.left:
                top_left_fragment = top_left_fragment.left
                continue
            if top_left_fragment.top:
                top_left_fragment = top_left_fragment.top
                continue
            break # a top-left fragment was found (ideally "the", but there might be "more than one" if incorrect configurations)

        # Attempt to reconstruct image
        reconstructed_image = [None for _ in range(16)]
        start_fragment = top_left_fragment # positioned at start of each row
        for i in range(4):
            if not start_fragment: # end of puzzle reached
                break
            current_fragment = start_fragment # traverses row
            for j in range(4):
                if not current_fragment: # end of row reached
                    break
                reconstructed_image[i * 4 + j] = current_fragment.id
                current_fragment = current_fragment.right
            start_fragment = start_fragment.bottom

        # Compare reconstructed image with correct image orientation
        random.seed(index)
        random.shuffle(reconstructed_image) # if in correct order, will be equal to [0, 1, 2, ..., 13, 14, 15]
        print(reconstructed_image)
        if reconstructed_image == list(range(16)): # successful reconstruction
            images_recreated += 1

except Exception as e:
    print(f"An error occurred: {e}")

# Display stats
print("\n==============================\nBORDERS CORRECTLY MATCHED:")
for correct, count in Counter(accuracies).most_common():
    print(f"{correct}/48: {count} ({(100 * count / len(entries)):.3f}%)")

print(("\nPUZZLES SUCCESSFULLY RECONSTRUCTED: "), end="")
print(f"{images_recreated}/{len(entries)} ({(100 * images_recreated / len(entries)):.3f}%)")