import ml_collections

# Define a Hyperparameter dictionary for easy experimentation and hyperparameter
# optimization


def model_config():
    cfg_dictionary = {
        "root_dir": "./Data",
        "relationships_file": "relationship.csv",
        "content_file": "content.csv",
        "test_file": "test.csv",
        "validation_split": 0.9,
        "epochs": 10,
        "batch_size": 256,
        "embedding_size": 256,
        "random_seed": 42,
        "model_checkpoint": "NCF99",
    }
    configs = ml_collections.FrozenConfigDict(cfg_dictionary)

    return configs


cfg = model_config()
