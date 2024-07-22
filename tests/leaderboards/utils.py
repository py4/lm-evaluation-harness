import json
import os
from typing import Dict, List


def filter_dict(d, key) -> Dict:
    """
    Filter a dictionary to include only the keys that contain the specified substring.

    Args:
        d (Dict): The dictionary to filter.
        key (str): The substring to filter the dictionary keys by.

    Returns:
        Dict: A dictionary containing only the keys that contain the specified substring.
    """
    return {k: v for k, v in d.items() if key in k}


class ParseConfig:
    """
    A class to parse a dictionary object into an attribute-based object.

    The `ParseConfig` class allows accessing dictionary keys as attributes and
    is designed to be used with dictionaries. It supports nested dictionaries and lists,
    recursively converting them into `ParseConfig` objects.

    Attributes:
        dict_obj (dict): The dictionary object to parse.

    Example:
        config_dict = {
            "transformers_version": "4.42.3",
            "params": {
                "model": "hf",
                "batch_size": 1
            },
        }

        config = ParseConfig(config_dict)

        # Accessing attributes
        print(config.transformers_version)
        >>> 4.42.3
        print(config.params.model)
        >>> hf
    """

    def __init__(self, dict_obj):
        """
        Initialize the ParseConfig object with the given dictionary object.

        Args:
            dict_obj (dict): The dictionary object to parse.
        """
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                setattr(self, key, ParseConfig(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [
                        ParseConfig(item) if isinstance(item, dict) else item
                        for item in value
                    ],
                )
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' is not defined in the config file.")

    def __getitem__(self, key):
        """
        Get the value of the key from the config file.
        """
        return self.__dict__[key]

    def to_dict(self):
        """
        Convert the ParseConfig object back into a dictionary.
        """
        return self.__dict__

    def keys(self):
        """
        Return the keys of the configuration object.
        """
        if hasattr(self, "__dict__"):
            return self.__dict__.keys()
        else:
            raise TypeError("Object is not a dictionary.")

    def items(self):
        """
        Return an iterator of the configuration object's items.
        """
        if hasattr(self, "__dict__"):
            return iter(self.__dict__.items())
        else:
            raise TypeError("Object is not a dictionary.")


def load_all_configs(device: str) -> List[Dict]:
    """
    Load all configuration files that include the specified device in their name.

    This function reads configuration files from the 'tests/leaderboards/testconfigs' directory.
    It filters the files to include only those that match the given device (e.g., 'cpu' or 'gpu').

    Args:
        device (str): The device type to filter configuration files by. Valid options are 'cpu' or 'gpu'.
                      If None, the default is 'cpu'.

    Returns:
        List[Dict]: A list of dictionaries containing the configuration data from the files.
    """

    if device is None:
        device = "cpu"

    valid_devices = {"cpu", "gpu"}
    if device not in valid_devices:
        raise ValueError(f"Invalid device {device}. Must be one of {valid_devices}.")

    configs_dir = os.path.join("tests", "leaderboards", "testconfigs")
    config_files = [
        f for f in os.listdir(configs_dir) if device in f and f.endswith(".json")
    ]

    data_list = []
    for filename in config_files:
        filepath = os.path.join(configs_dir, filename)
        with open(filepath, "r") as f:
            data_list.append(json.load(f))
    return data_list
