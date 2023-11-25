#!/bin/bash

python -m memory_profiler sample.py --init_from=gpt2 --start="To be or not to be, that is the question." --max_new_tokens=128 --num_samples=3

# python -m memory_profiler sample.py --init_from=gpt2 --start="To be or not to be, that is the question." --max_new_tokens=128 --num_samples=3