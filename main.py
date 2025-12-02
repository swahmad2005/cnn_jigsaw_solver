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

    # Scramble an image set
    user_input = input("Would you like to scramble a new set of images to test the CNN on? (y/n): ")
    if (user_input.lower() == "y"):
        images = input("Please enter the name of the directory containing these images (e.g. \"sample_images\"): ")
        print()
        scramble_images.scramble(images)
        print()
    
    # Unscramble an image set
    user_input = input("Would you like to unscramble a set of images with the CNN? (y/n): ")
    if (user_input.lower() == "y"):
        images = input("Please enter the name of the directory containing these images (e.g. \"sample_images_scrambled\"): ")
        print()
        unscramble_images.unscramble(images, model_name)
        print()
    
    # Test accuracy of unscrambled images
    user_input = input("Would you like to estimate the accuracy of the CNN's unscrambled images? (y/n): ")
    if (user_input.lower() == "y"):
        images = input("Please enter the name of the original directory containing these images, WITHOUT the suffix (e.g. \"sample_images\"; \"sample_images_scrambled_unscrambled\" would need to be present): ")
        print()
        test_accuracy.compare(images)
        print()

if __name__ == "__main__":
    main()