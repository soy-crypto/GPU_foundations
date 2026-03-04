from vllm import LLM, SamplingParams
import sys

model_name = sys.argv[1]

llm = LLM(model=model_name)

params = SamplingParams(
    max_tokens=50
)

outputs = llm.generate(
    "Explain GPU computing",
    params
)

for o in outputs:
    print(o.outputs[0].text)