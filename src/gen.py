"""Generate response using the Gemini model."""

import math
import os

import torch
from dotenv import load_dotenv
from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Set the number of Gemini API keys
KEY_NUMS = 6


load_dotenv()
API_KEYS = [os.getenv(f"API_KEY_{i + 1}") for i in range(KEY_NUMS)]
failed_keys = set()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(5))
def google_chat_completion_with_backoff(client, **kwargs):
    """Chat completion with backoff for Google API."""
    return client.models.generate_content(**kwargs)


def get_response_google(requests: list[str]) -> list[dict[str, str | float]]:
    """Get a response from the Gemini model using the first API key with quota."""
    #print("getting response")
    responses = []
    for request in requests:
        for key in API_KEYS:
            if key in failed_keys:
                continue
            client = genai.Client(api_key=key)
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=request,
                    config=genai.types.GenerateContentConfig(
                        temperature=0,
                    ),
                )
                responses.append(response.model_dump(exclude_unset=False))
                break
            except Exception as e:
                if "429 RESOURCE_EXHAUSTED" in str(e) and (
                    "'quotaValue': '1000'" in str(e) or "'quotaValue': '1500'" in str(e)
                ):
                    print(f"Error: {e}, trying next key...")
                    failed_keys.add(key)
                    continue
                else:
                    print(f"Server error {e}, retrying...")
                    response = google_chat_completion_with_backoff(
                        client=client,
                        model="gemini-2.0-flash",
                        contents=request,
                        config=genai.types.GenerateContentConfig(
                            temperature=0,
                        ),
                    )
                    responses.append(response.model_dump(exclude_unset=False))
                    break
        else:
            raise RuntimeError("All API keys exhausted or invalid.")
    return responses
