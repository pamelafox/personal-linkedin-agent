import asyncio
import logging
import os
import json
from enum import Enum
from typing import Any
from pydantic import BaseModel

import azure.identity
import openai
from dotenv import load_dotenv
from rich.logging import RichHandler

from playwright.async_api import async_playwright, Page

# Setup logging with rich
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("weekend_planner")
logger.setLevel(logging.INFO)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
if API_HOST == "github":
    client = openai.AsyncOpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.AzureDeveloperCliCredential(tenant_id=os.environ["AZURE_TENANT_ID"]),
        "https://cognitiveservices.azure.com/.default")
    client = openai.AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
elif API_HOST == "ollama":
    client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="none")
    MODEL_NAME = "llama3.1:latest"

class InvitationDecision(Enum):
    ACCEPT = "accept"
    IGNORE = "ignore"
    UNDECIDED = "undecided"

class MakeInvitationDecision(BaseModel):
    """Function tool to make a decision on LinkedIn invitations."""
    decision: InvitationDecision
    reason: str

async def ask_llm_for_decision(client, invitation_card, invitation_info: str):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system",
            "content": 
            """Decide whether to accept or ignore LinkedIn invitations based on the profile information provided.
Accept if the person has a technical role or mutual connections or works at Microsoft.
Ignore if they are a recruiter.
If you have any uncertainty at all as to whether the person meets the acceptance criteria, respond with 'undecided'.
"""},
            {"role": "user", "content": invitation_info},
        ],
        tools=[openai.pydantic_function_tool(MakeInvitationDecision)],
    )
    # add handling for tool_calls being none
    if not response.choices or not response.choices[0].message.tool_calls:
        logger.warning("No tool calls found in the response: %s", response.choices[0].message)
        # Sometimes the LLM will respond with just "accept" or "ignore" without tool calls
        try:
            decision_reason = "No reason provided by the agent."
            decision = InvitationDecision(response.choices[0].message.content.strip().lower())
        except ValueError:
            logger.warning("Invalid decision value: %s", response.choices[0].message.content)
            decision = InvitationDecision.UNDECIDED
    else:
        arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        decision = InvitationDecision(arguments["decision"])
        decision_reason = arguments["reason"]
    if decision == InvitationDecision.ACCEPT:
        accept_button = await invitation_card.query_selector("button[aria-label*='Accept']")
        if accept_button:
            await accept_button.click()
        else:
            logger.warning("Accept button not found on the card.")
            decision = InvitationDecision.UNDECIDED
    elif decision == InvitationDecision.IGNORE:
        ignore_button = await invitation_card.query_selector("button[aria-label*='Ignore']")
        if ignore_button:
            await ignore_button.click()
        else:
            logger.warning("Ignore button not found on the card.")
            decision = InvitationDecision.UNDECIDED
    return MakeInvitationDecision(decision=decision, reason=decision_reason)

async def get_profile_info(page: Page, profile_url: str) -> str:
    """Visit the profile page and extract relevant information."""
    # Open profile in a new tab
    new_page = await page.context.new_page()
    await new_page.goto(profile_url)
    await new_page.wait_for_load_state("load")
    
    # Just grab the whole main region
    main_content = await new_page.query_selector("main")
    if not main_content:
        logger.warning(f"Main content not found on profile page: {profile_url}")
        return "Profile information not available."
    profile_text = await main_content.inner_text()
    await new_page.close()
    return profile_text

async def process_linkedin_invitations(num_to_accept: int):
    logger.info("Starting LinkedIn invitation processing...")
    results = []
    processed_count = 0

    # Set up the storage state for Playwright
    os.makedirs("playwright/.auth", exist_ok=True)
    # If file doesn't exist, create it with empty JSON
    if not os.path.exists("playwright/.auth/state.json"):
        file_path = "playwright/.auth/state.json"
        with open(file_path, "w") as f:
            f.write("{}")
        f.close()

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state="playwright/.auth/state.json")
    
        # Create a new page
        page = await context.new_page()

        # Check if the user is logged in
        await page.goto("https://www.linkedin.com/feed")
        if not page.url.startswith("https://www.linkedin.com/feed"):
            logger.info("User is not logged in. Please log in manually...")
            await page.goto("https://www.linkedin.com/login")
            await page.wait_for_url("https://www.linkedin.com/feed/**", timeout=120000)
            logger.info("Login detected. Saving storage state...")
            await context.storage_state(path="playwright/.auth/state.json")

        logger.info("Navigating to invitation manager...")

        # Go to invitation manager
        await page.goto("https://www.linkedin.com/mynetwork/invitation-manager/")
        await page.wait_for_load_state("load")

        # Process invitations using the componentkey selectors from the screenshot
        await page.wait_for_selector("[componentkey='InvitationManagerPage_InvitationsList']")
        invitation_cards = await page.query_selector_all("[componentkey='InvitationManagerPage_InvitationsList'] > div[componentkey^='auto-component-']")
        logger.info(f"Found {len(invitation_cards)} initial invitation cards")

        # Initialize the last processed index
        last_processed_index = 0

        while processed_count < num_to_accept and len(invitation_cards) > 0:
            for index, card in enumerate(invitation_cards):
                # Skip cards that have already been processed
                if index < last_processed_index:
                    continue

                if processed_count >= num_to_accept:
                    break

                # Extract profile info
                name_element = await card.query_selector("a > strong")
                if not name_element:
                    continue
                name = await name_element.inner_text()

                # Get profile link
                profile_link_element = await card.query_selector("a")
                profile_link = await profile_link_element.get_attribute("href") if profile_link_element else "Unknown"
                if not profile_link.startswith("http"):
                    profile_link = f"https://www.linkedin.com{profile_link}"

                # Get job title
                job_title_element = await card.query_selector("p:nth-of-type(2)")
                job_title = await job_title_element.inner_text() if job_title_element else "Unknown"

                # Check for mutual connections
                connection_info_element = await card.query_selector("*:has-text('mutual connection')")
                connection_info = await connection_info_element.inner_text() if connection_info_element else ""
                has_mutual_connections = "mutual connection" in connection_info.lower()

                # Use the agent variable from the run function
                decision_message = f"Name: {name}, Job Title: {job_title}, Profile Link: {profile_link}, Connection Info: {connection_info}. Respond with 'accept', 'ignore', or 'undecided' and provide a reason for your decision."
                decision_and_reason = await ask_llm_for_decision(client, card, decision_message)

                # If agent is undecided, fetch more information from profile
                if decision_and_reason.decision == InvitationDecision.UNDECIDED:
                    logger.info(f"Agent is undecided about {name}. Fetching profile information...")
                    profile_info = await get_profile_info(page, profile_link)
                    
                    # Ask agent again with more context
                    detailed_message = f"Additional profile information for {name} ({job_title}):\n{profile_info}\n\nBased on this additional information, should we accept or ignore this invitation? Provide a reason for your decision."
                    decision_and_reason = await ask_llm_for_decision(client, card, detailed_message)
                    logger.info(f"Agent's final decision for {name}: {decision_and_reason.decision} - {decision_and_reason.reason}")

                # Store results
                results.append({"name": name, "profile": profile_link, "job_title": job_title, "mutual_connections": has_mutual_connections, "decision": decision_and_reason.decision, "reason": decision_and_reason.reason})

                processed_count += 1

                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

                if processed_count >= num_to_accept:
                    break

            # Update the last processed index
            last_processed_index = len(invitation_cards)

            if processed_count < num_to_accept:
                logger.info(f"Processed {processed_count} invitations so far. Scrolling to load more...")
                # Scroll down to load more
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)  # Wait for new cards to load

                # Get updated cards
                invitation_cards = await page.query_selector_all("[componentkey='InvitationManagerPage_InvitationsList'] > div[componentkey^='auto-component-']")

                if len(invitation_cards) == 0:
                    logger.info("No more invitations found")
                    break

        # Generate report
        logger.info("\n=== LinkedIn Invitation Processing Report ===")
        logger.info(f"Total invitations processed: {processed_count}")

        accepted = [r for r in results if r["decision"] == InvitationDecision.ACCEPT]
        ignored = [r for r in results if r["decision"] == InvitationDecision.IGNORE]
        undecided = [r for r in results if r["decision"] == InvitationDecision.UNDECIDED]

        logger.info(f"\nAccepted ({len(accepted)}):")
        for invitation in accepted:
            logger.info(f"- {invitation['name']} ({invitation['job_title']})")
            logger.info(f"  Profile: {invitation['profile']}")
            logger.info(f"  Mutual connections: {invitation['mutual_connections']}")

        logger.info(f"\nIgnored ({len(ignored)}):")
        for invitation in ignored:
            logger.info(f"- {invitation['name']} ({invitation['job_title']})")
            logger.info(f"  Profile: {invitation['profile']}")
            logger.info(f"  Reason: {invitation['reason']}")

        logger.info(f"\nUndecided ({len(undecided)}):")
        for invitation in undecided:
            logger.info(f"- {invitation['name']} ({invitation['job_title']}) {invitation['profile']}")
            logger.info(f"  Reason: {invitation['reason']}")

        # Close the browser
        await browser.close()

    return results

if __name__ == "__main__":
    asyncio.run(process_linkedin_invitations(30))
