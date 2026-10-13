"""Record a real agent that uses an open model through Hugging Face Inference Providers.

Needs a Hugging Face token (https://huggingface.co/settings/tokens) and the hf extra:

    pip install -e ".[hf]"
    $env:HF_TOKEN = "hf_..."          # PowerShell;  export HF_TOKEN=hf_... on macOS/Linux
    python examples/hf_quickstart.py
    agent-doctor runs

Each run makes two short LLM calls (a few hundred tokens), so it uses very little of the
free monthly credit. Change the model with the AD_MODEL environment variable.
"""

from __future__ import annotations

import os
import sys

from agent_doctor import instrument, trace_run, trace_tool

try:
    from huggingface_hub import InferenceClient
except ImportError:
    sys.exit('Install the Hugging Face client first: pip install -e ".[hf]"')

MODEL = os.environ.get("AD_MODEL", "Qwen/Qwen2.5-7B-Instruct")
if not os.environ.get("HF_TOKEN"):
    sys.exit("Set HF_TOKEN to your Hugging Face token first.")

client = instrument(InferenceClient(model=MODEL), agent="weather-bot")


@trace_tool
def get_weather(city: str) -> dict:
    """A stand-in weather tool, so the example needs no second API."""
    fake = {"delhi": (33, "haze"), "mumbai": (29, "humid"), "bengaluru": (24, "cloudy")}
    temp, sky = fake.get(city.strip().lower(), (26, "clear"))
    return {"city": city, "temp_c": temp, "sky": sky}


@trace_run(agent="weather-bot", prompt_version="v1")
def ask(question: str) -> str:
    city = client.chat_completion(
        messages=[
            {"role": "system", "content": "Reply with only the city name in the question."},
            {"role": "user", "content": question},
        ],
        max_tokens=10,
    ).choices[0].message.content.strip().strip(".")

    weather = get_weather(city)

    return client.chat_completion(
        messages=[
            {"role": "system", "content": "Answer in one friendly sentence."},
            {"role": "user", "content": f"{question}\nWeather data: {weather}"},
        ],
        max_tokens=80,
    ).choices[0].message.content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Do I need an umbrella in Mumbai today?"
    print(ask(q))
    print("\nRecorded. See it with: agent-doctor runs")
