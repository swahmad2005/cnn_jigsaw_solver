# 'scramble_images.py' contains the method to scramble a directory of images

import os
from PIL import Image
import random

def scramble(images):
    entries = os.listdir("./" + images + "/") # retrieve images
    if not os.path.isdir(images + "_scrambled/"): # create new folder for scrambled images (if required)
        os.mkdir(images + "_scrambled/")
    resized_images = False # issue warning if image sizes aren't 512x512
    
    for index, file in enumerate(entries):
        print(f"Scrambling... ({index + 1}/{len(entries)})")

        img = Image.open("./" + images + "/" + file)

        width, height = img.size
        if not width == 512 and not height == 512: # not of size 512, so resize
            resize_images = True
            img = img.resize((512, 512))
        
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
                scrambled.paste(piece, (j * 128, i * 128)) # paste to stitched image
        scrambled.save(images + "_scrambled/" + file) # save image

    print(f"Scrambled images saved to \"{images}_scrambled\" directory.")
    if resized_images:
        print("WARNING: some images were detected to not be of size 512x512, so they were rescaled.")
