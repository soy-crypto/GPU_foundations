MODEL=meta-llama/Llama-3.1-8B-Instruct

trtllm-build \
  --checkpoint_dir checkpoints/llama3 \
  --output_dir engines/llama3 \
  --dtype float16