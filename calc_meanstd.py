import os
import cv2
import numpy as np

train_path = "./dataset/train/" # path to train images

sum_means = (0, 0, 0)
sum_vars = (0, 0, 0)

try:
    train_images = os.listdir(train_path)

    # Iterate through train dataset
    for index, file in enumerate(train_images):
        print(f"({index + 1}/{len(train_images)})")
        img = cv2.imread(train_path + file)
        normalized_img = img.astype(np.float32) / 255.0 # scale values from 0 to 1
        mean, std = cv2.meanStdDev(normalized_img) # get mean and std
        sum_means += mean.flatten()
        sum_vars += std.flatten() ** 2
    print("Means:", sum_means / len(train_images))
    print("Standard Deviations:", (sum_vars / len(train_images)) ** 0.5)
        
except Exception as e:
   print(f"An error occurred: {e}")