import argparse
import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-src", default="./tokenizer_fast")
    p.add_argument("--out-dir", default="./eva-mini131k-eva_gpt-dense-fp32")
    p.add_argument("--orig-max-pos", type=int, default=8192)
    p.add_argument("--new-max-pos", type=int, default=131072)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--intermediate-size", type=int, default=8192)
    p.add_argument("--num-hidden-layers", type=int, default=16)
    p.add_argument("--num-attention-heads", type=int, default=32)
    p.add_argument("--num-key-value-heads", type=int, default=8)
    p.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="eager")
    args = p.parse_args()

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer_src, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.tokenizer_src, use_fast=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.bos_token

    cfg = AutoConfig.for_model("eva_gpt")
    cfg.vocab_size = len(tok)
    cfg.hidden_size = args.hidden_size
    cfg.intermediate_size = args.intermediate_size
    cfg.num_hidden_layers = args.num_hidden_layers
    cfg.num_attention_heads = args.num_attention_heads
    cfg.num_key_value_heads = args.num_key_value_heads
    cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    cfg.hidden_act = "silu"
    cfg.max_position_embeddings = args.new_max_pos
    cfg.sliding_window = args.orig_max_pos
    cfg.layer_types = ["sliding_attention", "sliding_attention", "full_attention"] * 5 + ["sliding_attention"]
    cfg.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": args.new_max_pos / args.orig_max_pos,
        "original_max_position_embeddings": args.orig_max_pos,
    }
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.pad_token_id = tok.pad_token_id
    cfg.num_local_experts = 1
    cfg.num_experts_per_tok = 1
    cfg.router_aux_loss_coef = 0.0
    cfg.output_router_logits = False
    cfg.tie_word_embeddings = True
    cfg.use_cache = False
    cfg._attn_implementation = args.attn_implementation

    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=args.attn_implementation)
    model.to(dtype=torch.float32, device="cpu")
    model.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size="100GB")
    cfg.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
