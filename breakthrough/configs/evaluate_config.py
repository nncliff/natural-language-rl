from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.input_path = "" # Data load path
    cfg.output_path = "" # Data save path
    cfg.model_path = "" # Model load path

    cfg.llm_sample_config = get_llm_sample_config()
    cfg.tensor_parallel_size = 1
    # Experiment note
    cfg.note = ""
    return cfg


def get_llm_sample_config():
    cfg = config_dict.ConfigDict()
    cfg.batch_size = 10000
    # LLm sampling params
    cfg.temperature = 1.0
    cfg.top_k = 50
    cfg.top_p = 0.95
    cfg.max_tokens = 512
    cfg.prompt_logprobs = False
    # num samples
    cfg.n = 1
    return cfg