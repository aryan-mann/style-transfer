from enum import Enum


class OUTPUT_TYPE(Enum):
    # Outputs from each iteration will be shown
    PROGRESSION = 1
    # Only the final output will be shown
    ONLY_FINAL = 2


class IMAGE_INITIALIZATION_SCHEME(Enum):
    # Initialize as random noise
    RANDOM_NOISE = 1
    # Initialise as the style image
    STYLE_IMAGE = 2
    # Initialise as the content image
    CONTENT_IMAGE = 3
    # Initialise as the average of the style and content image
    STYLE_CONTENT_MERGER = 4
