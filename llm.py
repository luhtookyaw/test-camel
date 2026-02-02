# llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Thin wrapper around OpenAI chat completion.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content

def call_llm_messages(
    messages: list,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Use this for multi-turn (history) calls.
    messages example:
    [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages
    )
    return response.choices[0].message.content