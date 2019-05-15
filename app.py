import sys
from keras.applications import vgg19
from keras import backend
import numpy as np
from matplotlib import pyplot as pp
from scipy.optimize import fmin_l_bfgs_b

####################
# Input Parameters #
####################
from image_transformation import to_vgg, from_vgg, initialize_vgg_transforms
from loss_functions import content_loss, style_loss, initialize_loss_functions

# The paths to our content and style image
content_image_path = 'images/pics/chicago.jpg'
style_image_path = 'images/styles/unknown.jpg'

# Resolve circular dependencies via initialization function
output_height, output_width = (512, 512)
initialize_vgg_transforms(output_height, output_width)
initialize_loss_functions(output_height, output_width)

#############################
# Style Transfer Parameters #
#############################
influence_of_content = 0.025
influence_of_style = 100

####################
# Other Parameters #
####################
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
feature_map = layer_name_to_output['block5_conv2']
content_image_features = feature_map[0, :, :, :]
guess_features = feature_map[2, :, :, :]
loss += influence_of_content * content_loss(content_image_features, guess_features)

# ********* ADDING THE CONTENT LOSS *********#
style_layers = ['block5_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for item in style_layers:
    feature_map = layer_name_to_output[item]
    style_features = feature_map[1, :, :, :]
    guess_features = feature_map[2, :, :, :]

    loss += (influence_of_style / len(style_layers)) * style_loss(style_features, guess_features)

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

    if len(step_results[1:]) == 1:
        cached_gradients = step_results[1].flatten().astype('float64')
    else:
        cached_gradients = np.array(step_results[1:]).flatten().astype('float64')

    return cached_loss


def get_cached_gradients(guensor):
    """
    Get the gradient of the latest 
    :param guensor:
    :return:
    """
    global cached_loss, cached_gradients
    gradient_to_return = cached_gradients
    cached_loss = None
    cached_gradients = None
    return gradient_to_return


# ****************************#
# RUN OPTIMIZATION ALGORITHM #
# ****************************#

# Set initial guess to a random image
#guess_image = to_vgg(np.random.random_integers(0, 255, (output_height, output_width, 3)), is_path=False)
# Set initial guess to the content image
guess_image = to_vgg(content_image_path, is_path=True)

grid_x, grid_y = (2, 2)
pp.figure()
for i in range(grid_x * grid_y):
    print('Start of iteration', (i+1))

    guess_image, iteration_loss, _ = fmin_l_bfgs_b(get_cached_loss, guess_image.flatten(),
                                                   fprime=get_cached_gradients, maxfun=10)

    print('Current loss value:', iteration_loss)

    pp.subplot(grid_x, grid_y, (i + 1))
    pp.imshow(from_vgg(guess_image.copy()))
    pp.title("Guess " + str(i+1))

pp.show()
