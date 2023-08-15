import os
import openai

from config import (
    CHAT_MODEL, 
    COMPLETIONS_MODEL, 
    API_KEY
)
code_request = open("tests/openai/messages/improve_code_efficiency.txt", "r").read()

openai.api_key = os.getenv("OPENAI_API_KEY")
messages = [
    {"role": "system", "content": "You will be provided with a piece of Python code, and your task is to provide ideas for efficiency improvements."},
    {"role": "user", "content": "code:\n```python\nfrom typing import List\n\ndef has_sum_k(nums: List[int], k: int) -> bool:\n    n = len(nums)\n    seen = set()\n    for num in nums:\n        if k - num in seen:\n            return True\n        seen.add(num)\n    return False\n```\n"}
]
response = openai.ChatCompletion.create(
  model=CHAT_MODEL,
  messages=messages,
  temperature=0,
  max_tokens=1024
)

print(response.choices[0].message['content'])
print("finish")
