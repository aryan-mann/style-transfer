import numpy as np
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

output_height = 512
output_width = 512


def initialize_vgg_transforms(height, width):
    """
    Set the output height and width for the transformations
    :param height: The height of the output transforms (in pixels)
    :param width: The width of the output transforms (in pixels)
    :return:
    """
    global output_width, output_height
    output_height = height
    output_width = width


def to_vgg(path_or_image, is_path=True):
    """
    Converts an image to a representation expected as the input by the VGG-19.
    :param path_or_image: Either an image of the shape (Height, Width, 3) or a path to one
    :param is_path: Whether path_or_image refers to an image or a path to an image
    :return: An image of the shape (1, Height, Width, 3)
    """
    if is_path:
        path_or_image = load_img(path_or_image, target_size=(output_height, output_width))

    # Convert the image to an array of shape (Height, Width, 3)
    path_or_image = img_to_array(path_or_image)

    # Add an extra dimension at the start so that the shape becomes (Batch Size, Height, Width, 3)
    path_or_image = np.expand_dims(path_or_image, axis=0)

    # Send it in for pre-processing for the VGG-19 Network
    path_or_image = vgg19.preprocess_input(path_or_image)
    return path_or_image


def from_vgg(vgg_input_image):
    """
    Converts a VGG-19 input image into a displayable image
    :param vgg_input_image: An image of shape (Height * Width * 3)
    :return: An image of shape (Height, Width, 3)
    """

    # Convert the image from a long array into a 2D matrix with 3 color channels
    vgg_input_image = vgg_input_image.reshape((output_height, output_width, 3))

    # Remove the mean-pixel, taken from GitHub issue <CITE>
    vgg_input_image[:, :, 0] += 103.939
    vgg_input_image[:, :, 1] += 116.779
    vgg_input_image[:, :, 2] += 123.68

    # My attempt at adding the mean-pixel back in
    # for i in [0, 1, 2]:
    #    vgg_input_image[:, :, i] += np.mean(vgg_input_image[:, :, i])

    # BGR -> RGB
    vgg_input_image = vgg_input_image[:, :, ::-1]

    # Make sure all the values are between 0 and 255
    vgg_input_image = np.clip(vgg_input_image, 0, 255).astype('uint8')
    return vgg_input_image


def get_random_image():
    return np.random.random_integers(0, 255, (output_height, output_width, 3))


def average_images(path_1, path_2):
    img_1 = load_img(path_1, target_size=(output_height, output_width))
    img_1 = img_to_array(img_1)
    img_2 = load_img(path_2, target_size=(output_height, output_width))
    img_2 = img_to_array(img_2)

    return np.clip((img_1 + img_2)/2, 0, 255).astype('uint8')
