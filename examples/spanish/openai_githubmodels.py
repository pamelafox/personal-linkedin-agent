import os

from openai import OpenAI

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente muy útil.",
        },
        {
            "role": "user",
            "content": "Cual es la capital de Ecuador?",
        },
    ],
    model=os.getenv("GITHUB_MODEL", "gpt-4o"),
)
print(response.choices[0].message.content)
