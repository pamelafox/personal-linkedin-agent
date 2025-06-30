import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

client = ChatCompletionsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)

response = client.complete(
    messages=[
        SystemMessage(content="Eres un asistente ùtil."),
        UserMessage(content="¿Cuál es la capital de Francia?"),
    ],
    model=os.getenv("GITHUB_MODEL", "gpt-4o"),
)
print(response.choices[0].message.content)
