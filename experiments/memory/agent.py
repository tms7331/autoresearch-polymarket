"""
Agent harness for the memory prediction system.

Uses the Claude Agent SDK (claude-agent-sdk) to run a Claude Code-powered agent
with a memory_lookup tool.  This uses your Claude Code subscription — no
separate API key needed.

The agent receives a market question, calls memory_lookup to retrieve relevant
memories, and produces a probability estimate.

This file is FIXED infrastructure — do not modify.
The experiment loop modifies model.py only.
"""

import json
import re

import anyio
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, query

# ---------------------------------------------------------------------------
# The memory model instance is set before each prediction run.
# ---------------------------------------------------------------------------

_memory_model = None


def set_memory_model(model):
    """Set the memory model instance that the tool will query."""
    global _memory_model
    _memory_model = model


# ---------------------------------------------------------------------------
# Memory lookup tool (exposed to the agent via MCP)
# ---------------------------------------------------------------------------

@tool(
    "memory_lookup",
    "Search the memory bank for relevant past experiences, news analyses, "
    "and market outcomes. Returns memories ranked by relevance and learned "
    "utility (Q-value). Each memory includes the original market question, "
    "category, outcome, and related news headlines.",
    {"query": str},
)
async def memory_lookup(args):
    q = args.get("query", "")
    if _memory_model is None:
        result = {"error": "Memory model not initialized"}
    else:
        result = _memory_model.tool_lookup(q)
    return {
        "content": [
            {"type": "text", "text": json.dumps(result, default=str)}
        ]
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

memory_server = create_sdk_mcp_server(
    name="memory",
    version="1.0.0",
    tools=[memory_lookup],
)

# ---------------------------------------------------------------------------
# Agent options
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a prediction market analyst. Given a market question, estimate the
probability (0.01 to 0.99) that it resolves "Yes".

You have a memory_lookup tool that searches a bank of past news analyses and
market outcomes. Use it to ground your estimate in evidence. You may call the
tool more than once with different queries if the first result is insufficient.

After consulting memory, respond with EXACTLY this JSON and nothing else:
{"probability": <float>, "confidence": <float 0-1>, "reasoning": "<1 sentence>"}
"""

AGENT_OPTIONS = ClaudeAgentOptions(
    system_prompt=SYSTEM_PROMPT,
    model="claude-haiku-4-5-20251001",
    max_turns=5,
    mcp_servers={"memory": memory_server},
    allowed_tools=["mcp__memory__memory_lookup"],
)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def _predict_market_async(question: str) -> dict:
    """Run the agent on a single market question."""
    prompt = (
        f"What is the probability that this market resolves Yes?\n\n"
        f"Market: {question}"
    )

    full_text = ""
    async for message in query(prompt=prompt, options=AGENT_OPTIONS):
        # AssistantMessage has .content list of TextBlocks
        if hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                if hasattr(block, "text"):
                    full_text += block.text
        elif hasattr(message, "content") and isinstance(message.content, str):
            full_text += message.content
        elif hasattr(message, "result") and message.result:
            full_text += str(message.result)

    return _parse_response(full_text)


def predict_market(question: str, memory_model) -> dict:
    """Synchronous wrapper — run the agent to predict a single market.

    Args:
        question: The market question (e.g. "Will X happen by Y?").
        memory_model: An object with ``tool_lookup(query) -> dict``
            and ``start_prediction(market_id)`` methods.

    Returns:
        dict with keys: probability (float), confidence (float), reasoning (str).
    """
    set_memory_model(memory_model)
    return anyio.run(_predict_market_async, question)


def _parse_response(text: str) -> dict:
    """Extract probability/confidence/reasoning from the agent's JSON reply."""
    try:
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "probability": max(0.01, min(0.99, float(data.get("probability", 0.5)))),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                "reasoning": str(data.get("reasoning", "")),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {"probability": 0.5, "confidence": 0.0, "reasoning": "parse_error"}
