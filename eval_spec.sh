#!/bin/bash

## confirm that kv cache on top of quantization is working
#python sample_spec.py --init_from=gpt2 --start="humaneval.jsonl" --batch_size=32 --max_new_tokens=128 --prompt_length=128 --num_samples=4 --num_warmup=1

# speculative run
python sample_spec.py --init_from=gpt2 --batch_size=1 --max_new_tokens=50 --prompt_length=1024 --num_samples=3 --num_warmup=1