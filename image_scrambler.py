from PIL import Image
import random

try:
    # Open base image (name of original non-split image)
    img_name = input("Enter image name: ")
    img = Image.open(img_name)
    img_name = img_name.split('.') # split file name and file extension for ease
    
    # Split image into 16 sections (first image piece is top left, complete by row)
    img_sections = [] # store all image sections
    for y in range(4): # 4 rows
        for x in range(4): # 4 cols
            piece = img.crop((x * 128, y * 128, (x + 1) * 128, (y + 1) * 128)) # crop section (image piece always 128x128)
            img_sections.append(piece)

    # Randomized permutation to scramble image
    permutation = list(range(16)) # refers to each image piece
    random.shuffle(permutation) # scramble
    
    # Re-stitch image based on permutation
    scrambled = Image.new("RGB", (512, 512)) # image is always 512x512 in this dataset
    for y in range(4): # 4 rows
        for x in range(4): # 4 cols
            piece = img_sections[permutation[y * 4 + x]] # retrieve random image piece
            scrambled.paste(piece, (x * 128, y * 128)) # paste to stitched image
    scrambled.save(f"{img_name[0]}_scrambled.{img_name[1]}")

except Exception as e:
    print(f"An error occurred: {e}")