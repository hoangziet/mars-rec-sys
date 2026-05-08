"""
configs.py
==========
Centralized hyperparameter configurations for all models.
"""

MODEL_CONFIGS = {
    "sasrec": {
        "model_kwargs": {
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "train_kwargs": {
            "batch_size": 256,
            "epochs": 20,
            "lr": 1e-3,
            "max_len": 50,
            "gradient_clip": 5.0,
        },
    },
    "gsasrec": {
        "model_kwargs": {
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "train_kwargs": {
            "batch_size": 256,
            "epochs": 20,
            "lr": 1e-3,
            "max_len": 50,
            "gradient_clip": 5.0,
        },
    },
    "gru4rec": {
        "model_kwargs": {
            "emb_dim": 64,
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.2,
        },
        "train_kwargs": {
            "batch_size": 512,
            "epochs": 20,
            "lr": 1e-3,
            "max_len": 50,
            "gradient_clip": 5.0,
        },
    },
    "bprmf": {
        "model_kwargs": {
            "emb_dim": 64,
        },
        "train_kwargs": {
            "batch_size": 1024,
            "epochs": 20,
            "lr": 1e-3,
            "reg_lambda": 1e-4,
            "max_len": 50,
            "gradient_clip": 0,
        },
    },
    "bert4rec": {
        "model_kwargs": {
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "train_kwargs": {
            "batch_size": 256,
            "epochs": 20,
            "lr": 1e-3,
            "max_len": 50,
            "gradient_clip": 5.0,
        },
    },
    "itemcf": {
        "model_kwargs": {
            "top_k_sim": 20,
        },
        "train_kwargs": {
            "max_len": 50,
            "num_neg": 99,
        },
    },
    "popularity": {
        "model_kwargs": {},
        "train_kwargs": {
            "max_len": 50,
            "num_neg": 99,
        },
    },
}

DEFAULT_SEED = 42
DEFAULT_DATA_DIR = "data/processed"
DEFAULT_OUTPUT_DIR = "experiments"
