# 'test_accuracy.py' tests how well the CNN performed in unscrambling images
# Compares each corresponding image between the unscrambled directory and the original directory
# Re-stitching images can change some pixel values by a little, but this method is still mostly reliable

import os
from PIL import Image, ImageChops, ImageStat
import random

def compare(images):
    try:
        original = os.listdir(images) # retrieve images
        unscrambled = os.listdir(images + "_unscrambled")

        num_correct = 0

        for i in range(len(original)):
            img1, img2 = Image.open(images + "/" + original[i]), Image.open(images + "_unscrambled/" + unscrambled[i])
            img1 = img1.resize((512, 512)) # resize in case not 512x512

            diff = ImageChops.difference(img1, img2) # perfectly identical will result in complete black image
            r, g, b = ImageStat.Stat(diff).extrema # minimum & maximum pixel values found in diferential image
            if r[1] - r[0] > 100 or g[1] - g[0] > 100 or b[1] - b[0] > 100: # 100 appears to give reliable results
                continue
            num_correct += 1 # images identified as resembling

            img1.close()
            img2.close()
        
        print(f"ESTIMATED NUMBER OF IMAGES SUCCESSFULLY RECONSTRUCTED: {num_correct}/{len(original)} ({(100 * num_correct / len(original)):.3f}%)")

    except Exception as e:
        print(f"An error occurred: {e}")