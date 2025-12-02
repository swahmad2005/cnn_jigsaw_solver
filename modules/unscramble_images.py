# 'unscramble_images.py' contains the method to unscramble a directory of scrambled images using a trained CNN model

import os
from PIL import Image
from modules import cnn_model
import torch

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
    
def unscramble(images, model_name):
    cnn_model.unscrambling_image = True # required for CNN to read single images

    entries = os.listdir("./" + images + "/") # retrieve images
    if not os.path.isdir(images + "_unscrambled/"): # create new folder for scrambled images (if required)
        os.mkdir(images + "_unscrambled/")

    for index, file in enumerate(entries):
        print(f"Unscrambling... ({index + 1}/{len(entries)})")

        img = Image.open("./" + images + "/" + file)

        # Split image into 16 quadrants
        quadrants = [] # contains all 16 quadrants
        for row in range(4):
            for col in range(4):
                quadrants.append(img.crop((col * 128, row * 128, col * 128 + 128, row * 128 + 128))) # each quadrant is 128x128

        # Create CNN inputs (all 960 possible border connections)
        model = cnn_model.CNN()
        model = torch.load(model_name)
        model.eval()
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
                        combined_border = Image.new("RGB", (2 * cnn_model.BORDER_PIXELS, 128))
                        combined_border.paste(quadrants[f2].crop((128 - cnn_model.BORDER_PIXELS, 0, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f1].crop((0, 0, cnn_model.BORDER_PIXELS, 128)), (cnn_model.BORDER_PIXELS, 0))
                    elif side == 1:
                        combined_border = Image.new("RGB", (2 * cnn_model.BORDER_PIXELS, 128))
                        combined_border.paste(quadrants[f1].crop((128 - cnn_model.BORDER_PIXELS, 0, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f2].crop((0, 0, cnn_model.BORDER_PIXELS, 128)), (cnn_model.BORDER_PIXELS, 0))
                    elif side == 2:
                        combined_border = Image.new("RGB", (128, 2 * cnn_model.BORDER_PIXELS))
                        combined_border.paste(quadrants[f2].crop((0, 128 - cnn_model.BORDER_PIXELS, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f1].crop((0, 0, 128, cnn_model.BORDER_PIXELS)), (0, cnn_model.BORDER_PIXELS))
                        combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
                    elif side == 3:
                        combined_border = Image.new("RGB", (128, 2 * cnn_model.BORDER_PIXELS))
                        combined_border.paste(quadrants[f1].crop((0, 128 - cnn_model.BORDER_PIXELS, 128, 128)), (0, 0))
                        combined_border.paste(quadrants[f2].crop((0, 0, 128, cnn_model.BORDER_PIXELS)), (0, cnn_model.BORDER_PIXELS))
                        combined_border = combined_border.transpose(Image.TRANSPOSE) # maintain 128x16 for consistency
                    outputs.append(model(cnn_model.transform(combined_border)).item())

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

        # Randomly guess unmatched fragments (model did a poor job already if this is the case)
        remaining = set(list(range(16))) - set(reconstructed_image)
        for i in range(16):
            if reconstructed_image[i] == None:
                reconstructed_image[i] = remaining.pop()

        # Re-stitch image and save
        unscrambled = Image.new("RGB", (512, 512))
        for i in range(4):
            for j in range(4):
                if reconstructed_image[i * 4 + j] == None:
                    continue
                piece = quadrants[reconstructed_image[i * 4 + j]] # retrieve random image piece
                unscrambled.paste(piece, (j * 128, i * 128)) # paste to stitched image
        unscrambled.save(images + "_unscrambled/" + file) # save image
    
    print(f"Unscrambled images saved to \"{images}_unscrambled\" directory.")