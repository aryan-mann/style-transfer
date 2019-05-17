import os
import re
import sys

import numpy as np
from keras import backend
from keras.applications import vgg19
from matplotlib import pyplot as pp
from scipy.optimize import fmin_l_bfgs_b

from enums import OUTPUT_TYPE, IMAGE_INITIALIZATION_SCHEME
from image_transformation import to_vgg, from_vgg, initialize_vgg_transforms, get_random_image, average_images
from loss_functions import content_loss, style_loss, initialize_loss_functions

####################
# META INFORMATION #
####################
loss_history = []

#############################
# Style Transfer Parameters #
#############################
influence_of_content = 0.
influence_of_style = 1

# The paths to our content and style image
content_image_path = 'images/pics/white_rose.jpg'
style_image_path = 'images/styles/waters_edge.jpg'
# Each iteration 'i' will be saved at: 'images/results/{output_prefix}/(i).{output_format}'
output_prefix = 'C2S2t'
output_format = 'png'

# How should the results of the training be shown.
# Reference "enums.py" to know more about the options
output_type = OUTPUT_TYPE.ONLY_FINAL

# How should we initialize our initial guess
guess_image_initialization_scheme = IMAGE_INITIALIZATION_SCHEME.CONTENT_IMAGE

# The layers to use for comparing style
content_layer = 'block5_conv2'
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]


def base_save_location():
    scheme_str = str(re.search('.+\.(.+)', str(guess_image_initialization_scheme)).group(1))
    return 'images/results/' + output_prefix + '__' + scheme_str.lower()


def iteration_save_location(i):
    return base_save_location() + '/(' + str(i) + ').' + output_format


# Create the directory for the output
if not os.path.exists(base_save_location()):
    os.makedirs(base_save_location())

####################
# INITIALIZE STUFF #
####################

# Resolve circular dependencies via initialization functions
output_height, output_width = (512, 512)
initialize_vgg_transforms(output_height, output_width)
initialize_loss_functions(output_height, output_width)

# Initialize our initial guess
guess_image = None
# Random Noise
if guess_image_initialization_scheme == IMAGE_INITIALIZATION_SCHEME.RANDOM_NOISE:
    guess_image = to_vgg(get_random_image(), is_path=False)
# Content Image
elif guess_image_initialization_scheme == IMAGE_INITIALIZATION_SCHEME.CONTENT_IMAGE:
    guess_image = to_vgg(content_image_path)
# Style Image
elif guess_image_initialization_scheme == IMAGE_INITIALIZATION_SCHEME.STYLE_IMAGE:
    guess_image = to_vgg(style_image_path)
# Average of Content and Style Image
elif guess_image_initialization_scheme == IMAGE_INITIALIZATION_SCHEME.STYLE_CONTENT_MERGER:
    guess_image = to_vgg(average_images(content_image_path, style_image_path))

should_run_optimizer = False if (sys.argv[1] == "False") else True

# Create content, style, and guess inputs for the VGG-Network
guess_tensor = backend.placeholder((1, output_height, output_width, 3))
content_image_tensor = backend.variable(to_vgg(content_image_path))
style_image_tensors = backend.variable(to_vgg(style_image_path))
# Put them together for easy processing into the VGG-network
# The added dimension at the start refers to the batch size
concatenated_input_tensor = backend.concatenate([
    content_image_tensor,
    style_image_tensors,
    guess_tensor
], axis=0)

# Initialize the VGG-19 Network with the input as the content, style, and guess and no classification layer
model = vgg19.VGG19(input_tensor=concatenated_input_tensor,
                    weights='imagenet', include_top=False)

# Create a Hashmap<String, Tensor> that takes in the layer name and spits out the output for that layer
layer_name_to_output = dict([(layer.name, layer.output) for layer in model.layers])

# ***************************** #
# CREATING THE BACKEND FUNCTION #
# ***************************** #
loss = backend.variable(0)

# ********* ADDING THE CONTENT LOSS *********#
feature_map = layer_name_to_output[content_layer]
content_image_features = feature_map[0, :, :, :]
guess_features = feature_map[2, :, :, :]
instance_content_loss = content_loss(content_image_features, guess_features)
loss += influence_of_content * instance_content_loss

# ********* ADDING THE STYLE LOSS *********#

instance_style_loss = 0
for item in style_layers:
    feature_map = layer_name_to_output[item]
    style_features = feature_map[1, :, :, :]
    guess_features = feature_map[2, :, :, :]
    layer_style_loss = (influence_of_style / len(style_layers)) * style_loss(style_features, guess_features)
    instance_style_loss += layer_style_loss

    loss += layer_style_loss

# *********  GRADIENT AND LOSS CALCULATION FOR SciPy Optimizer  *********#
guess_gradients = backend.gradients(loss, guess_tensor)
return_values = [loss]
return_values += guess_gradients

"""
    THE BACKEND FUNCTION
    -   Takes in the guess_tensor
    -   Calculates the content and style loss
    -   Calculates the gradient
    -   Returns the results in the shape: [ loss, gradients ]
"""
guess_to_loss = backend.function([guess_tensor], return_values)

# *******************************#
# GRADIENT AND LOSS CALCULATION #
# *******************************#
cached_loss, cached_gradients = None, None


def get_cached_loss(guensor):
    guensor = guensor.reshape((1, output_height, output_width, 3))
    step_results = guess_to_loss([guensor])

    global cached_loss, cached_gradients
    cached_loss = step_results[0]

    cached_gradients = step_results[1].flatten().astype('float64')

    return cached_loss


def get_cached_gradients(guensor):
    """
    Get the gradient of the latest 
    :param guensor: Not used
    :return: A matrix of size (output_height, output_width, 3) representing the gradient
    """
    global cached_loss, cached_gradients
    gradient_to_return = cached_gradients
    cached_loss = None
    cached_gradients = None
    return gradient_to_return


# ****************************#
#  RUN OPTIMIZATION ALGORITHM #
# ****************************#

# --------------- #
# DISPLAY OPTIONS #
# --------------- #
# Number of iterations run: grid_x * grid_y
#   If OUTPUT_TYPE.PROGRESSION, then each iteration will be
#   display in a grid of size grid_x by grid_y
grid_x, grid_y = (100, 1)
pp.figure()

print("Starting Optimization Algorithm")
for count in range(grid_x * grid_y):

    guess_image, iteration_loss, _ = fmin_l_bfgs_b(get_cached_loss, guess_image.flatten(),
                                                   fprime=get_cached_gradients, maxfun=20)

    print("Iteration ", count, " loss: ", iteration_loss)
    loss_history.append(iteration_loss)

    resultant_img = from_vgg(guess_image.copy())
    pp.imsave(iteration_save_location(count), resultant_img)

    if output_type is OUTPUT_TYPE.PROGRESSION:
        pp.subplot(grid_x, grid_y, (count + 1))
        pp.imshow(from_vgg(guess_image.copy()))
        pp.title("Guess " + str(count + 1))

if output_type is OUTPUT_TYPE.ONLY_FINAL:
    pp.imshow(from_vgg(guess_image.copy()))
    pp.title("Final Output (" + str(grid_x * grid_y) + " iterations)")

pp.figure()
pp.plot(loss_history)
pp.title("Loss per Iteration")
pp.xlabel("Iteration")
pp.ylabel("Total Loss Value")
pp.show()