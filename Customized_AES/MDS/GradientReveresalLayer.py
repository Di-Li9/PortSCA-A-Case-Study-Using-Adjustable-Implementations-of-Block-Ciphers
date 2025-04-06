import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.custom_gradient
def gradient_reversal(x, alpha=1.0):
    """Custom gradient reversal operation for adversarial training"""
    def grad(dy):
        # Reverse gradients and scale by alpha during backpropagation
        return -dy * alpha, None  # Second None is for alpha derivative (not used)
    return x, grad  # Return original input in forward pass

class GradientReversalLayer(Layer):
    """Custom Keras layer implementing Gradient Reversal for Domain Adaptation"""
    
    def __init__(self, **kwargs):
        """
        Initialize the Gradient Reversal Layer
        :param kwargs: Standard layer keyword arguments
        """
        super(GradientReversalLayer, self).__init__(**kwargs)

    def call(self, x, alpha=1.0):
        """
        Forward pass with gradient reversal
        :param x: Input tensor (features)
        :param alpha: Scaling factor for gradient reversal (default: 1.0)
        :return: Output tensor (same as input) with reversed gradients
        """
        # Apply gradient reversal operation during backward pass
        # Forward: identity function
        # Backward: gradient reversal with alpha scaling
        return gradient_reversal(x, alpha)