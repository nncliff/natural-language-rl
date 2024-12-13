from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.input_path = "" # Data load path
    cfg.output_path = "" # Data save path

    cfg.note = ""
    return cfg