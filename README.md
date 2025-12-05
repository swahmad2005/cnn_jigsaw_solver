## Getting Started  
First, clone the repository and import the Conda environment (ensure that you have Conda installed): `conda env create -f environment.yml`

Activate the Conda environment: `conda activate cnn_jigsaw_solver`

The file 'main.py' provides all of the needed functionality. Simply run this program: `python main.py`

## Steps of 'main.py'
For simplicity and ease-of-use, 'main.py' contains three key stages of interacting with the CNN, each of which are optional:
1. **Training**: This trains a new CNN model on the provided training dataset (i.e., 4500 images stored in 'dataset/train/'), and saves the PyTorch model to the current working directory as 'cnn_trained.pt'. You can skip this step and use the already provided model 'cnn_pretrained.pt'.
2. **Testing**: This tests how well CNN model performs on individual samples on the provided testing dataset (i.e., 1500 images stored in 'dataset/test/'). This step does not involve unscrambling images, it simply tests how frequently the CNN can correctly identify matching borders and non-matching borders.
3. **Scrambling/Unscrambling**: This step combines scrambling a provided image set (stored in a new directory ending in '_scrambled'), unscrambling the scrambled image set (stored in a new directory ending in '_unscrambled'), and then determining how well the CNN model did in unscrambling images.

## Creating your own set of images
Two image sets are already provided as examples ('sample_images/' and 'forest_images/'), which can be used in Step 3 of 'main.py'. You can also provide your own set of images in a similar manner to this. Images of size 512x512 are recommended, but in the event that this is not provided, the images will simply be rescaled to 512x512 (for that reason, rectangular images are even less recommended).

## Miscellaneous files (in 'misc/')
The 'misc/' folder contains files that are no longer necessary (they were used in earlier phases of the project), but they still contain working files that you can run:
- '**algorithmic_unscrambler.py**': The non-neural network approach of matching image borders and unscrambling images.
- '**calc_meanstd.py**': Calculates the mean and standard deviation of a directory of images (specifically, of the training dataset).
- '**cnn.py**': Performs training and testing of a CNN in one go.
- '**cnn_unscrambler.py**': Attempts to scramble/unscramble all 1500 images of the testing dataset using the pretrained CNN ('cnn_pretrained.pt' in the home directory) and measures the accuracy. Does this without creating any new directories.

