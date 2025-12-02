import os
from PIL import Image
import random

dataset = "./sample/"

try:
    entries = os.listdir(dataset) # retrieve images
    os.mkdir(dataset + "scrambled/") # create new folder for scrambled images

    # Perform tests on each image
    for index, file in enumerate(entries):
        print(f"Scrambling... ({index + 1}/{len(entries)})")

        name, extension = file.split(".")
        img = Image.open(dataset + file)
        print(file)
        
        # Split image into 16 sections (first image piece is top left, complete by row)
        quadrants = [] # contains all 16 quadrants
        for row in range(4):
            for col in range(4):
                quadrants.append(img.crop((col * 128, row * 128, col * 128 + 128, row * 128 + 128))) # each quadrant is 128x128

        # Randomized permutation to scramble image
        permutation = list(range(16)) # refers to each image piece
        random.shuffle(permutation) # scramble
    
        # Re-stitch image based on permutation
        scrambled = Image.new("RGB", (512, 512)) # image is always 512x512 in this dataset
        for i in range(4): # 4 rows
            for j in range(4): # 4 cols
                piece = quadrants[permutation[i * 4 + j]] # retrieve random image piece
                scrambled.paste(piece, (i * 128, j * 128)) # paste to stitched image
        scrambled.save(dataset + "scrambled/" + file) # save image

except Exception as e:
    print(f"An error occurred: {e}")