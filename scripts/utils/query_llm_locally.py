"""
Simple CLI script to query Qwen3-4B with a system prompt and user prompt.
Usage:
    python query_llm_locally.py --system-prompt "You are a helpful assistant." --user-prompt "Hello!"
    python query_llm_locally.py --system-prompt-file system.txt --user-prompt-file user.txt
"""

import argparse
import logging
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


MODEL_NAME = "Qwen/Qwen3-4B"


def load_model(model_name: str):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def query(
    tokenizer, model, system_prompt: str, user_prompt: str, max_new_tokens: int = 1024
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Qwen3-4B locally with 8-bit quantization."
    )
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
    )
    parser.add_argument("--user-prompt", type=str, default=None)
    parser.add_argument(
        "--user-prompt-file",
        type=str,
        default="prompts/er_reason_xf_examples_user_prompt.txt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.system_prompt_file:
        with open(args.system_prompt_file, "r") as f:
            system_prompt = f.read().strip()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        raise ValueError("Provide --system-prompt or --system-prompt-file")

    if args.user_prompt_file:
        with open(args.user_prompt_file, "r") as f:
            user_prompt = f.read().strip()
    elif args.user_prompt:
        user_prompt = args.user_prompt
    else:
        raise ValueError("Provide --user-prompt or --user-prompt-file")

    print(f"Loading {args.model_name} in 8-bit...")
    tokenizer, model = load_model(args.model_name)

    print("Generating...\n")
    response = query(tokenizer, model, system_prompt, user_prompt, args.max_new_tokens)
    print(response)


if __name__ == "__main__":
    main()
