import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Activation
from tensorflow.keras.layers import BatchNormalization, Concatenate, Softmax, AveragePooling1D

def build_feature_extractor():
    """
    Constructs the feature extraction subnetwork
    Architecture:
    - 1D Convolution → Batch Normalization → Average Pooling → Flatten
    Design Choices:
    - He Uniform initialization for better convergence with SELU activation
    - Average Pooling instead of Max Pooling for smoother feature extraction
    """
    model = tf.keras.Sequential([
        Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', 
              padding='same', name='block1_conv1'),
        BatchNormalization(),  # Stabilizes training through normalization
        AveragePooling1D(2, strides=2, name='block1_pool'),  # Reduces spatial dimensions
        Flatten()  # Prepares for dense layers
    ])
    return model

def build_label_classify_extractor():
    """
    Builds label classification head
    Structure:
    - Two 64-unit dense layers → 256-class softmax output
    Configuration:
    - SELU activation with He initialization maintains self-normalizing properties
    - Final layer size 256 corresponds to byte value classification (0-255)
    """
    model = tf.keras.Sequential([
        Dense(64, kernel_initializer='he_uniform', activation='selu', name='fc1'),
        Dense(64, kernel_initializer='he_uniform', activation='selu', name='fc2'),
        Dense(256, activation='softmax', name='predictions')  # Byte classification
    ])
    return model

def build_domain_classify_extractor():
    """
    Constructs domain classifier for domain adaptation
    Architecture:
    - Two 64-unit dense layers → Dropout → Binary classification
    Regularization:
    - 50% Dropout prevents overfitting to domain-specific features
    - Final 2-unit output for source/target domain discrimination
    """
    model = tf.keras.Sequential([
        Dense(64, kernel_initializer='he_uniform', activation='selu', name='pre_fc1'),
        Dense(64, kernel_initializer='he_uniform', activation='selu', name='pre_fc2'),
        tf.keras.layers.Dropout(0.5),  # Strong regularization for domain invariance
        Dense(2, activation='softmax', name="domain_cls_pred")  # Domain classification
    ])
    return model