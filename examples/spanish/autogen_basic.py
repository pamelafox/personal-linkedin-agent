import asyncio
import os

import azure.identity
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from dotenv import load_dotenv

# Setup the client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
if API_HOST == "github":
    client = OpenAIChatCompletionClient(model=os.getenv("GITHUB_MODEL", "gpt-4o"), api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAIChatCompletionClient(model=os.environ["AZURE_OPENAI_CHAT_MODEL"], api_version=os.environ["AZURE_OPENAI_VERSION"], azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_ad_token_provider=token_provider)


agent = AssistantAgent(
    "english_tutor", 
    model_client=client,
    system_message=(
        "Eres un tutor de inglés. Ayuda al usuario a aprender inglés. "
        "Traducir solo lenguaje natural al español informal latinoamericano. "
        "SOLO RESPONDE en español informal latinoamericano."
    ),
)


async def main() -> None:
    response = await agent.on_messages(
        [TextMessage(content="hola como estas?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)


if __name__ == "__main__":
    asyncio.run(main())
