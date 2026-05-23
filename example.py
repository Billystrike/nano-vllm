import os

import torch._dynamo
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

torch._dynamo.config.suppress_errors = True

def main():
    path = os.path.expanduser("/home/bstrike/huggingface/models--Qwen--Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=False,
        tensor_parallel_size=1,
        prefill_chunk_size=512,
        max_num_batched_tokens=16384,
    )

    sampling_params = SamplingParams(temperature=0.3, max_tokens=256)
    # long_context = "\n".join(["- 解释一次 chunked prefill 的好处。"] * 200)
    prompts = [
        "请用简洁的语言介绍一下 triton 的用途。\n", #+ long_context,
        "请给出两数之和的高效解法，并说明复杂度。\n" #+ long_context,
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    prompt_lens = [len(tokenizer.encode(prompt)) for prompt in prompts]
    print(f"Prompt token counts: {prompt_lens}, prefill_chunk_size=512")
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
