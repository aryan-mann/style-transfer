import app
from app import model as icn
from matplotlib import pyplot as pp
from keras import backend
import numpy as np

"""
    Testing out pre-processing and de-processing images
"-""
"""
# Process and load in the image
content_img = app.to_vgg('images/pics/chicago.jpg')
style_img = app.to_vgg('images/styles/unknown.jpg')
guess_img = app.to_vgg('images/styles/starry_night.jpg')

app.from_vgg(content_img)
"""
# The shape should be: (1, X, Y, 3)
print(processed_img.shape)
# It should look weird
pp.figure()
pp.imshow(processed_img[0, :, :, :])
pp.show()
# Deprocess it
deprocessed_img = app.vgg_image_to_display(processed_img)
# The shape should be: (X, Y, 3)
print(deprocessed_img.shape)
# It should look normal
pp.figure()
pp.imshow(deprocessed_img)
pp.show()

"-""
    Testing the outputs from each layer in the VGG network

def get_output_from_layer(layer_name):
    session = backend.get_session()
    return (session.run(app_2.layer_to_output[layer_name], feed_dict={app_2.model.input: together_img}))[0]

pp.figure()
figure_index = 1
total_figures = len(app_2.layer_to_output)

for name in app_2.layer_to_output:
    if name == "input_1":
        continue
    pp.subplot(total_figures, 1, figure_index)
    session = backend.get_session()
    output = session.run(app_2.layer_to_output[name], feed_dict={app_2.model.input: together_img})
    print(output[0, :, :, 0])
    figure_index += 1

pp.show()
"""