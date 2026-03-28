# import os
# from dotenv import load_dotenv
# from google.genai import Client

# # Load environment variables from .env
# load_dotenv()

# def generate_insight(prompt: str) -> str:
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY not set")

#     client = Client(api_key=api_key)

#     response = client.responses.generate(
#         model="gemini-1.5-t",
#         temperature=0.2,
#         max_output_tokens=300,
#         prompt=prompt
#     )

#     # Extract text from structured response
#     return response.output[0].content[0].text

# from gpt4all import GPT4All

# model = GPT4All("ggml-gpt4all-j-v1.3-groovy")

# def generate_insight(prompt: str) -> str:
#     response = model.generate(prompt, max_tokens=300)
#     return response

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# MODEL_NAME = "tiiuae/falcon-7b-instruct"  # completely public

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

# def generate_insight(prompt: str) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=300)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"


def generate_insight(prompt: str) -> str:
    """
    Call the Anthropic Claude API to generate an insight from a prompt.
    Raises ValueError if the API key is not set.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"]
