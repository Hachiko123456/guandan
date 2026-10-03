# Main/Subagent Execution Protocol

## Main agent

The main agent owns requirements, integration, Git, and acceptance. For every subagent result it must inspect the exact diff, reject out-of-scope files, run focused tests, run the stage acceptance command, run the full test suite, compare with the requirements, and only then update `STATUS.yaml`.

## Subagents

- Use disjoint write sets.
- Do not edit `STATUS.yaml` or mark stages accepted.
- Do not broaden scope.
- Report exact files, commands, tests, assumptions, and gaps.
- Do not use or copy FableDan.

## Status meanings

- `not_started`: no accepted implementation;
- `in_progress`: active implementation/review;
- `verified`: implementation/tests pass, main review pending;
- `accepted`: all gates and main review pass;
- `blocked`: concrete repeated blocker;
- `rejected`: reviewed implementation failed its gate.

A pytest pass is not the same as supervisory acceptance.

## Failure protocol

When returning work, the main agent gives the failing test, violated requirement, expected behavior, allowed files, and exact re-test command. The same subagent is reused where practical.
