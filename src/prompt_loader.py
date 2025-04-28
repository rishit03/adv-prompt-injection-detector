# src/prompt_loader.py

import json
from typing import List

def load_prompts(path: str) -> List[str]:
    with open(path, 'r') as f:
        return json.load(f)
