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

#limit_images = 0

accuracies = [] # tracks border comparison accuracies
images_recreated = 0 # images successfully reconstructed

try:
    entries = os.listdir(path) # retrieve images

    # Perform tests on each image
    for img_num, img_name in enumerate(entries):
        print(f"Testing image ({img_num + 1}/{len(entries)})")
        img = Image.open(path + img_name) # open image
        
        # Retrieve all border pixels for each image fragment
        borders = [] # represents all fragments' pixel borders on each side (16x4x128x3)
        for fragment in range(16): # iterate through each fragment
            fragment_border = []
            for side in range(4): # iterate through all 4 sides of fragment
                side_border = []
                for pixel in range(128): # retrieve each pixel
                    if side == 0: # left
                        side_border.append(img.getpixel((fragment // 4 * 128 + pixel, fragment % 4 * 128)))
                    elif side == 1: # right
                        side_border.append(img.getpixel((fragment // 4 * 128 + pixel, fragment % 4 * 128 + 127)))
                    elif side == 2: # top
                        side_border.append(img.getpixel((fragment // 4 * 128, fragment % 4 * 128 + pixel)))
                    elif side == 3: # bottom
                        side_border.append(img.getpixel((fragment // 4 * 128 + 127, fragment % 4 * 128 + pixel)))
                fragment_border.append(side_border)
            borders.append(fragment_border)

        # Scramble image
        permutation = [i for i in range(16)]
        random.seed(img_num) # set new random seed (seed is image number for simplicity and consistency)
        random.shuffle(permutation)
        random.seed(img_num) # same seed
        random.shuffle(borders)

        # Compare all borders and find losses
        border_losses = [] # represents how much each border matches with each other (16x4x16)
        for f1 in range(16): # first fragment
            fragment_border_loss = []
            for side in range(4): # compare each side
                side_loss = []
                for f2 in range(16): # second fragment
                    if f1 == f2: # prevent comparing same fragment to itself
                        side_loss.append(float('inf'))
                    elif side == 0: # left border (compare to right border)
                        side_loss.append(border_loss(borders[f1][0], borders[f2][1]))
                    elif side == 1: # right border (compare to left border)
                        side_loss.append(border_loss(borders[f1][1], borders[f2][0]))
                    elif side == 2: # top border (compare to bottom border)
                        side_loss.append(border_loss(borders[f1][2], borders[f2][3]))
                    elif side == 3: # bottom border (compare to top border)
                        side_loss.append(border_loss(borders[f1][3], borders[f2][2]))
                #fragment_border_loss.append(side_loss)
                fragment_border_loss.append(confidence_dist(side_loss))
            border_losses.append(fragment_border_loss)
        
        # Test border match accuracies (Note: since there are only 48 matching borders, these are the ones tested here)
        num_correct = 0
        predicted_borders = [] # the borders predicted for each side of each fragment (16x4); each element is a tuple of: (loss, piece)
        for fragment in range(16):
            predicted_sides = []
            for side in range(4):
                min_loss = min(border_losses[fragment][side]) # minimum loss found for a side
                min_piece = border_losses[fragment][side].index(min_loss) # expected matching piece
                predicted_sides.append((min_loss, min_piece))
                if side == 0:
                    if permutation[fragment] % 4 == 0: # piece already on left border
                        continue
                    if permutation[min_piece] == permutation[fragment] - 1: # correct piece was guessed (the right border of left piece)
                        num_correct += 1
                elif side == 1:
                    if permutation[fragment] % 4 == 3: # piece already on right border
                        continue
                    if permutation[min_piece] == permutation[fragment] + 1: # correct piece was guessed (the left border of right piece)
                        num_correct += 1
                elif side == 2:
                    if permutation[fragment] <= 3: # piece already on top border
                        continue
                    if permutation[min_piece] == permutation[fragment] - 4: # correct piece was guessed (the bottom border of top piece)
                        num_correct += 1
                elif side == 3:
                    if permutation[fragment] >= 12: # piece already on bottom border
                        continue
                    if permutation[min_piece] == permutation[fragment] + 4: # correct piece was guessed (the top border of bottom piece)
                        num_correct += 1
            predicted_borders.append(predicted_sides)
        accuracies.append(num_correct)

        # Attach most resembling fragments (top 48)
        reconstructed_fragments = [Fragment(i, None, None, None, None) for i in range(16)]
        for _ in range(48): # 48 matching borders (top 48 most resembling borders)
            # Find the minimum loss (piece to be placed next)
            min_fragment, min_side, min_loss = None, None, float('inf') # track minimum fragment/side based on minimum loss
            for fragment in range(16):
                for side in range(4):
                    if (predicted_borders[fragment][side][0] < min_loss): # new min_loss found
                        min_fragment, min_side, min_loss = fragment, side, predicted_borders[fragment][side][0]
            predicted_fragment = predicted_borders[min_fragment][min_side][1]
            #print(f"{_}: {min_loss, min_fragment, min_side, predicted_fragment}")

            predicted_borders[min_fragment][min_side] = (float('inf'), None) # remove the predicted piece as it now will be placed

            # Place the predicted fragment in the corresponding location (based on fragment/side of minimum loss)
            if min_side == 0: # best comparison found for a left border
                if reconstructed_fragments[min_fragment].left == None and reconstructed_fragments[predicted_fragment].right == None:
                    reconstructed_fragments[min_fragment].left = reconstructed_fragments[predicted_fragment]
                    reconstructed_fragments[predicted_fragment].right = reconstructed_fragments[min_fragment]
            elif min_side == 1: # best comparison found for a right border
                if reconstructed_fragments[min_fragment].right == None and reconstructed_fragments[predicted_fragment].left == None:
                    reconstructed_fragments[min_fragment].right = reconstructed_fragments[predicted_fragment]
                    reconstructed_fragments[predicted_fragment].left = reconstructed_fragments[min_fragment]
            elif min_side == 2: # best comparison found for a top border
                if reconstructed_fragments[min_fragment].top == None and reconstructed_fragments[predicted_fragment].bottom == None:
                    reconstructed_fragments[min_fragment].top = reconstructed_fragments[predicted_fragment]
                    reconstructed_fragments[predicted_fragment].bottom = reconstructed_fragments[min_fragment]
            elif min_side == 3: # best comparison found for a bottom border
                if reconstructed_fragments[min_fragment].bottom == None and reconstructed_fragments[predicted_fragment].top == None:
                    reconstructed_fragments[min_fragment].bottom = reconstructed_fragments[predicted_fragment]
                    reconstructed_fragments[predicted_fragment].top = reconstructed_fragments[min_fragment]
        
        #for fragment in reconstructed_fragments:
            #print(fragment.show_neighbors())

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

        # Compare reconstructed image with correct image
        random.seed(img_num)
        random.shuffle(reconstructed_image) # if in correct order, will be equal to [0, 1, 2, ..., 13, 14, 15]
        random.seed(img_num)
        random.shuffle(reconstructed_image) # if in correct order, will be equal to permutation
        if reconstructed_image == permutation: # successful reconstruction
            images_recreated += 1
        
        #limit_images += 1
        #if limit_images >= 10:
            #break

except Exception as e:
    print(f"An error occurred: {e}")

# Display stats
print("\n==============================\nBORDERS CORRECTLY MATCHED:")
for correct, count in Counter(accuracies).most_common():
    print(f"{correct}/48: {count} ({(100 * count / len(entries)):.3f}%)")

print(("\nPUZZLES SUCCESSFULLY RECONSTRUCTED: "), end="")
print(f"{images_recreated}/{len(entries)} ({(100 * images_recreated / len(entries)):.3f}%)")

# EXPECTED OUTPUTS:
#
# Default:
'''
BORDERS CORRECTLY MATCHED:
48/48: 1353 (90.200%)
47/48: 75 (5.000%)
46/48: 33 (2.200%)
45/48: 20 (1.333%)
44/48: 9 (0.600%)
42/48: 4 (0.267%)
43/48: 3 (0.200%)
37/48: 2 (0.133%)
34/48: 1 (0.067%)

PUZZLES SUCCESSFULLY RECONSTRUCTED: 767/1500 (51.133%)
'''
# Without 'confidence_dist':
'''
BORDERS CORRECTLY MATCHED:
48/48: 1353 (90.200%)
47/48: 75 (5.000%)
46/48: 33 (2.200%)
45/48: 20 (1.333%)
44/48: 9 (0.600%)
42/48: 4 (0.267%)
43/48: 3 (0.200%)
37/48: 2 (0.133%)
34/48: 1 (0.067%)

PUZZLES SUCCESSFULLY RECONSTRUCTED: 725/1500 (48.333%)
'''
# Sum of differences of border losses (simpler)
'''
BORDERS CORRECTLY MATCHED:
48/48: 1310 (87.333%)
47/48: 98 (6.533%)
46/48: 43 (2.867%)
45/48: 18 (1.200%)
44/48: 17 (1.133%)
42/48: 6 (0.400%)
43/48: 3 (0.200%)
41/48: 2 (0.133%)
34/48: 1 (0.067%)
36/48: 1 (0.067%)
37/48: 1 (0.067%)

PUZZLES SUCCESSFULLY RECONSTRUCTED: 718/1500 (47.867%)
'''