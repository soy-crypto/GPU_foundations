import time
import sys
from tensorrt_llm.runtime import ModelRunner
from tensorrt_llm.runtime import SamplingConfig

engine_dir = "engines/llama3"

prompt = "Explain GPU computing in one paragraph."

runner = ModelRunner.from_dir(engine_dir)

sampling = SamplingConfig(
    max_new_tokens=50,
    temperature=0.7
)

start = time.time()

outputs = runner.generate(
    [prompt],
    sampling_config=sampling
)

end = time.time()

print(outputs[0])

print("Latency:", end-start)