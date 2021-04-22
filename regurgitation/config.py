import yaml
import os
from collections import OrderedDict, abc
current_folder = os.path.split(os.path.abspath(__file__))[0]

class Config:
    def __new__(cls, arg={}):
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg
 
    def __init__(self, config_dict={}):
        self.config_dict = dict(config_dict)

    def __getattr__(self, name):
        return Config(self.config_dict[name])

    def __str__(self):
        return(self.config_dict.__str__())

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key != 'config_dict':
            self.config_dict[key] = value

    def export(self, output_path):
        with open(output_path, 'w') as f:
            yaml.dump(self.config_dict, f, default_flow_style=False)


def get_config(yaml_file):
    try:
        with open(yaml_file, 'r') as y:
            config_dict = yaml.load(y)
        config = Config(config_dict)
        print(config.config_dict)
    except:
        pass
    return(config)
