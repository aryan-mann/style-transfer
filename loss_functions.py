from keras import backend

# ********************#
# THE LOSS FUNCTIONS #
# ********************#
output_height, output_width = (512, 512)


def initialize_loss_functions(height, width):
    global output_height, output_width
    output_height = height
    output_width = width


def content_loss(content, guess):
    """
    The content loss between the content image and our guess image
    :param content: The feature map corresponding to the Content Image
    :param guess: The feature map corresponding to the Guess Image
    :return: A scalar value referring to the loss in content
    """
    return backend.sum(backend.square(guess - content))


def style_loss(style, guess):
    """
    The style loss between the style image and our guess image
    :param style:
    :param guess:
    :return:  A scalar value referring to the loss in style
    """

    def gram_matrix(matrix):
        """
        The gram matrix is defined as the product of a matrix and its transpose
        :param matrix: A n-Dimensional matrix
        :return: The gram matrix of the input matrix
        """
        features = backend.batch_flatten(backend.permute_dimensions(matrix, (2, 0, 1)))
        return backend.dot(features, backend.transpose(features))

    # 4 . N^2 . M^2 i.e. 4 * (Channels ^ 2) * (Size ^ 2)
    denominator_term = (4.0 * (3 ** 2) * ((output_height * output_width) ** 2))

    return backend.sum(backend.square(gram_matrix(style) - gram_matrix(guess))) / denominator_term