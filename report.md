# Decision Memo

Selected framework: Google Agent Development Kit (ADK).

## Options Considered
1. [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/), a developer- and code-first framework for orchestrating multi-agent workflows with structured state and tool orchestration, and a model-agnostic design optimized for Gemini.
2. [Simular Agent S3](https://www.simular.ai/articles/agent-s3), a computer-use framework aimed at scalable, high-performance GUI task completion via an Agent-Computer Interface.

## Decision
ADK is used to structure the Supervisor–Planner–Executor workflow for a three-role mobile QA system:
- Planner generates a structured JSON action plan from the natural language test cases
- Executor executes that plan deterministically with ADB and UI XML targeting
- Supervisor judges the outcome against expected behavior and reports a structured verdict (PASS/FAIL + failure type)

## Why this framework fits the challenge
The goal is a QA harness: deterministic execution, structured outputs, and auditable traces. That requires explicit separation between planning, execution, and evaluation, plus a reliable way to distinguish `FAILED_STEP` (couldn’t complete an intended interaction, e.g., element not found / navigation blocked / stuck) from `FAILED_ASSERTION` (reached the intended screen but the expected condition was not satisfied, e.g., a state/value/color mismatch).

ADK is a good fit because it is code-first and makes orchestration explicit. That matters here because the system must be inspectable: every tool call, retry, UI refresh, and decision should be traceable. ADK naturally supports structured inputs and outputs, which makes it straightforward to constrain and validate JSON plans and verdicts and to attach artifacts like UI dumps and screenshots to specific steps.

This project also benefits from a deterministic Executor. Android UI automation is sensitive to timing, overlays, stale UI trees, and permission prompts. Keeping device interaction deterministic reduces run-to-run variance and makes reliability engineering practical. The LLM is used where it adds the most value: the Planner interprets the test cases into an executable plan, and the Supervisor evaluates the final state and produces an actionable failure report.

Agent S3 targets high throughput, general purpose GUI autonomy, but its perception-driven loops can change actions with small timing shifts, overlays, or screen interpretation noise. This increases run-to-run variance and makes step-level failure attribution less stable. For this challenge, a deterministic test harness with consistent traces is the priority, so ADK is a better match.

## Risks and mitigations
ADK APIs may evolve, so integration is intentionally kept thin behind a small adapter layer, allowing the core QA logic to remain framework-agnostic. LLM variability is mitigated with structured outputs, deterministic guards for known flows, and artifact capture to support auditing and debugging.
