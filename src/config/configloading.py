import yaml

def load_config(path, default_path=None):
    """ Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

# 将cfg_special中的内容加到cfg中
def update_recursive(dict1, dict2):
    """ Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        # if v is still a dict, call the function update_recursive() recursively
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        # if v is not a dictionary, Assign the value of v to dict1[k]
        else:
            dict1[k] = v