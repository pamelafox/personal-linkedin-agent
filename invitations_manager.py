import argparse
import asyncio
import logging
import os
from enum import Enum
from pathlib import Path

import azure.identity
import yaml
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.logging import RichHandler

from playwright.async_api import ElementHandle, Page, async_playwright

# Setup logging with rich
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_level=True)])
logger = logging.getLogger("invitations_manager")
logger.setLevel(logging.INFO)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
    model = OpenAIChatModel(os.getenv("GITHUB_MODEL", "gpt-4o"), provider=OpenAIProvider(openai_client=client))
    logger.info("Using GitHub Models with model %s", model.model_name)
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    model = OpenAIChatModel(os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], provider=OpenAIProvider(openai_client=client))
    logger.info("Using Azure OpenAI with model %s", model.model_name)
else:
    raise ValueError(f"Unsupported API_HOST: {API_HOST}")


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
    decision: InvitationDecision | None = None


agent = Agent(
    model,
    system_prompt="""Decide whether to accept or ignore LinkedIn invitations based on the profile information provided.
Accept if the person has a technical role or mutual connections or works at Microsoft.
Ignore if they are a recruiter.
If you have any uncertainty at all as to whether the person meets the acceptance criteria, respond with 'undecided'.""",
    output_type=NativeOutput(InvitationDecision),
)


async def run_and_log_agent(case_name: str, input_message: str):
    """
    Run the agent on the input_message, log input/output as a YAML evals case, and return the decision.
    """
    log_path = "linkedin_invitation_cases.yaml"
    agent_result = await agent.run(input_message)
    decision = agent_result.output
    logger.info("%d input tokens, %d output tokens used for decision", agent_result.usage().input_tokens, agent_result.usage().output_tokens)
    case = {
        "name": case_name,
        "inputs": input_message,
        "expected_output": {
            "action": decision.action.value if decision else None,
            "reason": decision.reason if decision else None,
        }
        if decision
        else None,
        "metadata": {},
    }
    out_path = Path(log_path)
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            dataset_dict = yaml.safe_load(f) or {"cases": []}
    else:
        dataset_dict = {"cases": []}
    dataset_dict["cases"].append(case)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_dict, f, sort_keys=False, allow_unicode=True)
    return decision


async def get_invitation_info(card) -> Invitation | None:
    # Extract profile info
    name_element = await card.query_selector("a > strong")
    if not name_element:
        return None
    name = (await name_element.inner_text()).strip()

    # Get profile link
    profile_link_element = await card.query_selector("a")
    profile_link = await profile_link_element.get_attribute("href") if profile_link_element else "Unknown"
    if not profile_link.startswith("http"):
        profile_link = f"https://www.linkedin.com{profile_link}"

    # Get job title (single selector assumption)
    job_title_element = await card.query_selector("p:nth-of-type(2)")
    job_title = (await job_title_element.inner_text()).strip() if job_title_element else "Unknown"

    # Check for mutual connections (single phrase)
    connection_info_element = await card.query_selector("*:has-text('mutual connection')")
    connection_info = (await connection_info_element.inner_text()).strip() if connection_info_element else ""
    has_mutual_connections = "mutual connection" in connection_info.lower()
    return Invitation(name=name, profile=profile_link, job_title=job_title, mutual_connections=has_mutual_connections)


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


async def execute_action(card: ElementHandle, decision: InvitationDecision) -> InvitationDecision:
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


async def process_linkedin_invitations(num_to_process: int, record_eval_cases: bool = False):
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

        async def get_invitation_cards() -> list[ElementHandle]:
            primary_selector = "div[role='main'] div[componentkey^='auto-component-']:has(button[aria-label*='Accept'])"
            cards = await page.query_selector_all(primary_selector)
            if not cards:
                html_snippet = (await page.content())[:2000]
                logger.warning("No invitation cards found with primary selector. HTML snippet (2k chars): %s", html_snippet)
            return cards

        # Wait for the main region to be present
        await page.wait_for_selector("div[role='main']")
        invitation_cards = await get_invitation_cards()
        logger.info(f"Found {len(invitation_cards)} initial invitation card candidates")

        # New approach: track processed profile URLs and detect progress by appearance of any new unprocessed cards.
        processed_profiles: set[str] = set()
        consecutive_no_new_scrolls = 0
        max_no_new_scrolls = 5  # fail-safe to avoid infinite loop

        while processed_count < num_to_process:
            # Always fetch the current set of visible cards (DOM changes remove accepted/ignored ones)
            invitation_cards = await get_invitation_cards()
            if not invitation_cards:
                logger.info("No invitation cards currently visible.")
                break

            new_card_processed_this_pass = False

            for card in invitation_cards:
                if processed_count >= num_to_process:
                    break

                invitation = await get_invitation_info(card)
                if not invitation:
                    continue

                # Skip duplicates (already processed or re-rendered after accept/ignore)
                if invitation.profile in processed_profiles:
                    continue

                decision_message = f"Name: {invitation.name}, Job Title: {invitation.job_title}, " f"Profile Link: {invitation.profile}, Connection Info: {invitation.mutual_connections}."

                if record_eval_cases:
                    decision = await run_and_log_agent(invitation.name, decision_message)
                else:
                    agent_result = await agent.run(decision_message)
                    decision = agent_result.output
                    logger.debug(
                        "%d input tokens, %d output tokens used for decision",
                        agent_result.usage().input_tokens,
                        agent_result.usage().output_tokens,
                    )
                decision = await execute_action(card, decision)

                # If agent is undecided, fetch more information from profile
                if decision.action == InvitationAction.UNDECIDED:
                    logger.info(f"Agent is undecided about {invitation.name}. Fetching profile information...")
                    profile_info = await get_profile_info(page, invitation.profile)
                    detailed_message = f"Full profile information for {invitation.name} ({invitation.job_title}):\n{profile_info}\n\n" "Based on this additional information, should we accept or ignore this invitation? Provide a reason for your decision."
                    if record_eval_cases:
                        decision = await run_and_log_agent(invitation.name, detailed_message)
                    else:
                        detailed_result = await agent.run(detailed_message)
                        decision = detailed_result.output
                        logger.debug(
                            "%d input tokens, %d output tokens used for decision",
                            detailed_result.usage().input_tokens,
                            detailed_result.usage().output_tokens,
                        )
                    decision = await execute_action(card, decision)
                    logger.info(f"Agent's final decision for {invitation.name}: {decision.action} - {getattr(decision, 'reason', '')}")

                invitation.decision = decision
                results.append(invitation)
                processed_profiles.add(invitation.profile)
                processed_count += 1
                new_card_processed_this_pass = True

                # Small delay to avoid rate limiting / allow DOM to settle
                await asyncio.sleep(1)

            if processed_count >= num_to_process:
                break

            if not new_card_processed_this_pass:
                consecutive_no_new_scrolls += 1
                logger.info(f"No new unprocessed invitations visible (attempt {consecutive_no_new_scrolls}/{max_no_new_scrolls}). Scrolling for more...")
                await page.evaluate("window.scrollBy(0, window.innerHeight * 0.9)")
                await asyncio.sleep(2)
                if consecutive_no_new_scrolls >= max_no_new_scrolls:
                    logger.info("Reached maximum scroll attempts without discovering new invitations. Stopping.")
                    break
            else:
                # Reset counter if we made progress this pass
                consecutive_no_new_scrolls = 0
                logger.info(f"Processed {processed_count} invitations so far. Continuing to look for more until we reach {num_to_process}.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LinkedIn invitations.")
    parser.add_argument("--num-to-process", type=int, default=10, help="Number of LinkedIn invitations to process (default: 10).")
    parser.add_argument("--record-eval-cases", action="store_true", help="Record eval cases to YAML file.")
    args = parser.parse_args()

    asyncio.run(process_linkedin_invitations(args.num_to_process, record_eval_cases=args.record_eval_cases))
