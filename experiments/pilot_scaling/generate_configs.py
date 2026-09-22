"""
Generate OpenNMT-py configuration files for the 6 scaling benchmark arms:
- Arm A: 2E1D d128 (Baseline)
- Arm B: 4E4D d128 (Reproduce large paper)
- Arm C: 6E6D d128 (Reproduce largest paper)
- Arm D: 4E1D d128 (Asymmetric deep encoder)
- Arm E: 2E2D d128 (Extra decoder depth)
- Arm F: 2E1D d256 (Wide representation)
"""

import json
from pathlib import Path

ARMS = {
    "arm_a_2e1d_d128": {
        "name": "Arm A: 2E1D d128 (Baseline)",
        "enc_layers": 2,
        "dec_layers": 1,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 2048,
        "heads": 8,
        "notes": "Current production baseline with ultra-low latency."
    },
    "arm_b_4e4d_d128": {
        "name": "Arm B: 4E4D d128 (Medium Deep)",
        "enc_layers": 4,
        "dec_layers": 4,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 2048,
        "heads": 8,
        "notes": "Balanced deep encoder-decoder from sequence repair literature."
    },
    "arm_c_6e6d_d128": {
        "name": "Arm C: 6E6D d128 (Standard Deep)",
        "enc_layers": 6,
        "dec_layers": 6,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 2048,
        "heads": 8,
        "notes": "Standard Vaswani depth for maximum representation capacity."
    },
    "arm_d_4e1d_d128": {
        "name": "Arm D: 4E1D d128 (Asymmetric Deep Enc)",
        "enc_layers": 4,
        "dec_layers": 1,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 2048,
        "heads": 8,
        "notes": "Deep encoder (O(1) parallel) + shallow decoder (O(N) fast autoregressive)."
    },
    "arm_e_2e2d_d128": {
        "name": "Arm E: 2E2D d128 (Shallow Deep Dec)",
        "enc_layers": 2,
        "dec_layers": 2,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 2048,
        "heads": 8,
        "notes": "Tests whether adding a second decoder layer provides noticeable gain."
    },
    "arm_f_2e1d_d256": {
        "name": "Arm F: 2E1D d256 (Wide Representation)",
        "enc_layers": 2,
        "dec_layers": 1,
        "hidden_size": 256,
        "word_vec_size": 256,
        "transformer_ff": 2048,
        "heads": 8,
        "notes": "Tests depth vs width: shallow 2E1D architecture with doubled vector dimension."
    },
}

def make_config(arm_key, arm_spec, base_data_dir="data/base_v3_pilot", ckpt_root="checkpoints/pilot_scaling"):
    return {
        "save_data": f"{base_data_dir}/vocab",
        "src_vocab": f"{base_data_dir}/vocab.src",
        "tgt_vocab": f"{base_data_dir}/vocab.tgt",
        "src_vocab_size": 12000,
        "tgt_vocab_size": 12000,
        "overwrite": True,
        "data": {
            "corpus_1": {
                "path_src": f"{base_data_dir}/train.src",
                "path_tgt": f"{base_data_dir}/train.tgt",
                "transforms": ["sentencepiece"],
                "weight": 1
            },
            "valid": {
                "path_src": f"{base_data_dir}/valid.src",
                "path_tgt": f"{base_data_dir}/valid.tgt",
                "transforms": ["sentencepiece"]
            }
        },
        "src_subword_model": "data/tokenizer_v3/tokenizer.model",
        "tgt_subword_model": "data/tokenizer_v3/tokenizer.model",
        "save_model": f"{ckpt_root}/{arm_key}/model",
        "encoder_type": "transformer",
        "decoder_type": "transformer",
        "enc_layers": arm_spec["enc_layers"],
        "dec_layers": arm_spec["dec_layers"],
        "heads": arm_spec["heads"],
        "hidden_size": arm_spec["hidden_size"],
        "word_vec_size": arm_spec["word_vec_size"],
        "transformer_ff": arm_spec["transformer_ff"],
        "position_encoding": True,
        "self_attn_type": "scaled-dot",
        "dropout": [0.1],
        "attention_dropout": [0.1],
        "model_dtype": "fp16",
        "param_init": 0.0,
        "param_init_glorot": True,
        "optim": "adam",
        "learning_rate": 1.0,
        "adam_beta1": 0.8,
        "adam_beta2": 0.998,
        "adam_eps": 1e-08,
        "decay_method": "noam",
        "warmup_steps": 2000,
        "max_grad_norm": 1.0,
        "batch_type": "tokens",
        "batch_size": 32768,
        "bucket_size": 65536,
        "num_workers": 4,
        "normalization": "tokens",
        "report_every": 1000,
        "train_steps": 10000,
        "valid_steps": 1000,
        "save_checkpoint_steps": 5000,
        "keep_checkpoint": 2,
        "seed": 2026,
        "world_size": 1,
        "gpu_ranks": [0]
    }

def main():
    root = Path(__file__).resolve().parent.parent.parent
    config_dir = root / "experiments" / "pilot_scaling" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    for arm_key, arm_spec in ARMS.items():
        cfg = make_config(arm_key, arm_spec)
        out_path = config_dir / f"{arm_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        generated.append(out_path)
        print(f"Generated {arm_key}.json -> Enc: {arm_spec['enc_layers']}, Dec: {arm_spec['dec_layers']}, Dim: {arm_spec['hidden_size']}")
        
    print(f"\nAll {len(generated)} configs successfully written to {config_dir}!")

if __name__ == "__main__":
    main()
