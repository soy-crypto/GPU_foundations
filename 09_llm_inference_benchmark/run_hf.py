import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain GPU computing in one paragraph."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(
    **inputs,
    max_new_tokens=50
)

print(tokenizer.decode(output[0]))