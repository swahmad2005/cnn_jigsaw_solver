from modules import cnn_model, scramble_images, unscramble_images, test_accuracy
CNN = cnn_model.CNN # must redefine CNN class (bug)?

def invalid_input():
    print("ERROR: invalid input")
    exit(1) # exit program due to invalid input

def main():
    print("-----------------\nCNN JIGSAW SOLVER\n-----------------")

    model_name = "cnn_trained.pt" # default name after training; can be changed by user

    # TRAINING PHASE
    user_input = input("Would you like to train a new CNN model? (y/n): ") # anything other than 'y' is assumed 'n'
    if (user_input.lower() == "y"):
        print()
        cnn_model.train_model()
        print()
    else:
        model_name = input("Please enter the name of a pretrained CNN model (e.g. \"cnn_pretrained.pt\"): ")
    
    # TESTING PHASE
    user_input = input("Would you like to test the CNN model? (y/n): ")
    if (user_input.lower() == "y"):
        print()
        cnn_model.test_model(model_name)
        print()

    # Scramble/unscramble an image set
    images = input("UNSCRAMBLE TEST - Enter the name of a directory containing images (e.g. \"sample_images\"): ")
    scramble_images.scramble(images) # created scrambled directory of images
    print()
    unscramble_images.unscramble(images, model_name) # unscramble the scrambled directory of images
    print()
    test_accuracy.compare(images) # check how many images the CNN successfully unscrambled
    print()

if __name__ == "__main__":
    main()