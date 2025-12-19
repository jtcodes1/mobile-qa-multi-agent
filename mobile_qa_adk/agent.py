import os

try:
    from google.adk.agents import Agent
except Exception:
    # Compatibility with older ADK versions
    from google.adk.agents.llm_agent import Agent


def _ensure_gemini_key():
    if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def list_test_cases() -> dict:
    """ADK tool: return the names of available test cases."""
    import main

    return {"test_cases": [t["name"] for t in main.TESTS]}


def run_test_case(name: str) -> dict:
    """ADK tool: run a single test via main.run_one."""
    _ensure_gemini_key()
    import main

    run_dir, result = main.run_one(name)
    return {
        "status": "completed",
        "run_directory": run_dir,
        "result": result,
    }


def run_challenge_suite() -> dict:
    """ADK tool: run the entire suite via main.run_suite."""
    """
    Execute the full Obsidian mobile QA test suite using the existing
    Planner–Executor–Supervisor system.

    This function acts as an ADK tool and delegates all low-level device
    interaction and reasoning to the custom QA agents.
    """

    # Normalize API key naming between ADK and custom Gemini client
    _ensure_gemini_key()

    import main  # existing entrypoint

    run_dir, results = main.run_suite()
    return {
        "status": "completed",
        "run_directory": run_dir,
        "results": results,
    }


root_agent = Agent(
    name="mobile_qa_root_agent",
    description=(
        "ADK orchestration agent for mobile QA automation. "
        "Delegates planning, execution, and verification to a custom "
        "Planner–Executor–Supervisor system operating on an Android emulator."
    ),
    instruction=(
        "You are responsible for orchestrating execution of the mobile QA test suite. "
        "When instructed to run tests, invoke the list_test_cases, run_test_case, "
        "or run_challenge_suite tool as appropriate and return the resulting "
        "run directory and execution status."
    ),
    tools=[list_test_cases, run_test_case, run_challenge_suite],
)