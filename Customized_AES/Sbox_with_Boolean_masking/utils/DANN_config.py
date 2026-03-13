import os

class config(object):
    """Configuration class for managing model hyperparameters and paths"""
    
    # Default configuration values (note intentional spelling in variable name)
    __defualt_dict__ = {
        "byte": 0,  # Byte-related parameter placeholder
        "pre_model_path": None,  # Path to pre-trained model weights
        "checkpoints_dir": os.path.abspath("./checkpoints"),  # Default save directory
        "input_shape": (700, 1),  # Input dimension for the model
        "init_learning_rate": 3e-2,  # Initial learning rate for optimizer
        "momentum_rate": 0.9,  # Momentum parameter for optimizer
        "batch_size": 8000,  # Training batch size
        "epoch": 50,  # Number of training epochs
    }

    def __init__(self, **kwargs):
        """
        Initialize configuration with default values and custom overrides
        :param kwargs: Key-value pairs for configuration parameters
        """
        # Set default values first
        self.__dict__.update(self.__defualt_dict__)
        # Apply custom configurations
        self.__dict__.update(kwargs)

        # Create checkpoints directory if it doesn't exist
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

    def set(self, **kwargs):
        """
        Update configuration parameters after initialization
        :param kwargs: Key-value pairs to update configuration
        """
        # Merge new parameters into existing configuration
        self.__dict__.update(kwargs)

    def save_config(self, time):
        """
        Save current configuration to file with timestamp
        :param time: Timestamp string for versioning
        """
        # Update directory path with timestamp
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, time)
        
        # Create directory structure if needed
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

        # Write configuration to text file
        config_txt_path = os.path.join(self.config_dir, "config.txt")
        with open(config_txt_path, 'a') as f:
            for key, value in self.__dict__.items():
                # Handle special path formatting
                if key in ["checkpoints_dir"]:
                    value = os.path.join(value, time)
                # Write key-value pair to file
                f.write(f"{key}: {value}\n")