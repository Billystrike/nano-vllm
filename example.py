import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True # 遇到编译错误自动回退到 eager模式(用来Debug添加的)

def main():
    path = os.path.expanduser("/home/bstrike/huggingface/models--Qwen--Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.3, max_tokens=256)
    prompts = [
        "介绍一下什么是triton",
        "力扣第一题两数之和有什么较好的解法",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
