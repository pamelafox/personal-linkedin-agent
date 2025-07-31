from pathlib import Path

from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from invitations_manager import InvitationDecision, agent

# Define test cases for the LinkedIn invitation agent


class CorrectDecisionEvaluator(Evaluator[str, InvitationDecision]):
    def evaluate(self, ctx: EvaluatorContext[str, InvitationDecision]) -> float:
        if ctx.expected_output and ctx.output.action == ctx.expected_output.action:
            return 1.0
        else:
            return 0.0


# Load cases from YAML file
dataset = Dataset[str, InvitationDecision, dict].from_file(Path("linkedin_invitation_cases.yaml"))
dataset.evaluators = [IsInstance(type_name="InvitationDecision"), CorrectDecisionEvaluator()]


async def agent_decision(input_str: str) -> InvitationDecision:
    result = await agent.run(input_str)
    return result.output


def main():
    report = dataset.evaluate_sync(agent_decision)
    report.print(include_input=True, include_output=True, include_durations=False)


if __name__ == "__main__":
    main()
