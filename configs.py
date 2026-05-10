"""
configs.py
==========
Centralised hyperparameter configurations for all models.
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
            "epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "beta2": 0.98,
            "max_len": 50,
            "gradient_clip": 5.0,
            "early_stop_patience": 10,
            "early_stop_min_delta": 1e-4,
        },
    },
    "gsasrec": {
        "model_kwargs": {
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.2,
            "t": 0.5,      # gBCE temperature (paper default: 0.5)
            "num_neg": 32, # negatives per positive (paper default: 32)
            "pos_smoothing": 0.0,
        },
        "train_kwargs": {
            "batch_size": 256,
            "epochs": 50,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "beta2": 0.98,
            "max_len": 50,
            "num_neg": 32,  # passed to TrainSequenceDataset
            "gradient_clip": 5.0,
            "early_stop_patience": 10,
            "early_stop_min_delta": 1e-4,
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
            "loss_type": "ce",
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
            "mask_ratio": 0.2,
            "warmup_steps": 1000,
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
