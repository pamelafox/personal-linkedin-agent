import asyncio
import logging
import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel

import azure.identity
from dotenv import load_dotenv
from rich.logging import RichHandler
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from playwright.async_api import async_playwright, Page

# Setup logging with rich
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_level=True)])
logger = logging.getLogger("invitations_manager")
logger.setLevel(logging.INFO)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
    model = OpenAIModel(os.getenv("GITHUB_MODEL", "gpt-4o"), provider=OpenAIProvider(openai_client=client))
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    model = OpenAIModel(os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], provider=OpenAIProvider(openai_client=client))


class InvitationAction(Enum):
    ACCEPT = "accept"
    IGNORE = "ignore"
    UNDECIDED = "undecided"

class InvitationDecision(BaseModel):
    action: InvitationAction
    reason: str

class Invitation(BaseModel):
    name: str
    profile: str
    job_title: str
    mutual_connections: bool
    decision: Optional[InvitationDecision] = None


agent = Agent(
    model,
    system_prompt="""Decide whether to accept or ignore LinkedIn invitations based on the profile information provided.
Accept if the person has a technical role or mutual connections or works at Microsoft.
Ignore if they are a recruiter.
If you have any uncertainty at all as to whether the person meets the acceptance criteria, respond with 'undecided'.""",
    output_type=NativeOutput(InvitationDecision)
)

async def get_invitation_info(card) -> Optional[Invitation]:
    # Extract profile info
    name_element = await card.query_selector("a > strong")
    if not name_element:
        return None
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
    return Invitation(
        name=name,
        profile=profile_link,
        job_title=job_title,
        mutual_connections=has_mutual_connections
    )


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

async def execute_action(card: Page, decision: InvitationDecision) -> InvitationDecision:
    if decision.action == InvitationAction.ACCEPT:
        accept_button = await card.query_selector("button[aria-label*='Accept']")
        if accept_button:
            await accept_button.click()
        else:
            logger.warning("Accept button not found on the card.")
            decision.action = InvitationAction.UNDECIDED
    elif decision.action == InvitationAction.IGNORE:
        ignore_button = await card.query_selector("button[aria-label*='Ignore']")
        if ignore_button:
            await ignore_button.click()
        else:
            logger.warning("Ignore button not found on the card.")
            decision.action = InvitationAction.UNDECIDED
    return decision
            
            
async def process_linkedin_invitations(num_to_process: int):

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
            # Wait for the user to log in...
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

        while processed_count < num_to_process and len(invitation_cards) > 0:
            for index, card in enumerate(invitation_cards):
                # Skip cards that have already been processed
                if index < last_processed_index:
                    continue

                if processed_count >= num_to_process:
                    break
                
                invitation = await get_invitation_info(card)
                if not invitation:
                    continue

                decision_message = f"Name: {invitation.name}, Job Title: {invitation.job_title}, Profile Link: {invitation.profile}, Connection Info: {invitation.mutual_connections}."
                agent_result = await agent.run(decision_message)
                decision = agent_result.output
                decision = await execute_action(card, decision)

                # If agent is undecided, fetch more information from profile
                if decision.action == InvitationAction.UNDECIDED:
                    logger.info(f"Agent is undecided about {invitation.name}. Fetching profile information...")
                    profile_info = await get_profile_info(page, invitation.profile)

                    # Ask agent again with more context
                    detailed_message = f"Full profile information for {invitation.name} ({invitation.job_title}):\n{profile_info}\n\nBased on this additional information, should we accept or ignore this invitation? Provide a reason for your decision."
                    detailed_result = await agent.run(detailed_message)
                    decision = detailed_result.output
                    decision = await execute_action(card, decision)
                    logger.info(f"Agent's final decision for {invitation.name}: {decision.action} - {decision.reason}")

                invitation.decision = decision
                results.append(invitation)
                processed_count += 1

                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

                if processed_count >= num_to_process:
                    break

            # Update the last processed index
            last_processed_index = len(invitation_cards)

            if processed_count < num_to_process:
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

        accepted = [r for r in results if r.decision and r.decision.action == InvitationAction.ACCEPT]
        ignored = [r for r in results if r.decision and r.decision.action == InvitationAction.IGNORE]
        undecided = [r for r in results if r.decision and r.decision.action == InvitationAction.UNDECIDED]

        logger.info(f"\nAccepted ({len(accepted)}):")
        for invitation in accepted:
            logger.info(f"- {invitation.name} ({invitation.job_title})\n  Profile: {invitation.profile}\n  Mutual connections: {invitation.mutual_connections}")

        logger.info(f"\nIgnored ({len(ignored)}):")
        for invitation in ignored:
            logger.info(f"- {invitation.name} ({invitation.job_title})\n  Profile: {invitation.profile}\n  Reason: {invitation.decision.reason}")

        logger.info(f"\nUndecided ({len(undecided)}):")
        for invitation in undecided:
            logger.info(f"- {invitation.name} ({invitation.job_title}) {invitation.profile}\n  Reason: {invitation.decision.reason}")

        # Close the browser
        await browser.close()

    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process LinkedIn invitations.")
    parser.add_argument(
        "--num-to-process",
        type=int,
        default=2,
        help="Number of LinkedIn invitations to process (default: 2)."
    )
    args = parser.parse_args()
    
    asyncio.run(process_linkedin_invitations(args.num_to_process))
