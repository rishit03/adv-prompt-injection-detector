# src/utils.py

import re

def clean_prompt(prompt: str) -> str:
    # Simple preprocessing: trim whitespace and normalize case
    return prompt.strip().lower()
