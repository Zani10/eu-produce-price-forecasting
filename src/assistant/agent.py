# The buyer assistant: a language model that can call our tools.
#
# ask(question, use_tools=True)  -> the model can call the 5 tools (max 5 rounds), then answers
# ask(question, use_tools=False) -> same model and same instructions, but no tools
#
# The API key is read from the .env file (OPENAI_API_KEY). The key is never written anywhere else.

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.assistant.tools import SCHEMAS, TOOLS

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = "gpt-4.1-mini-2025-04-14"
MAX_ROUNDS = 5
MAX_TOKENS = 400

SYSTEM = """You are an assistant for fresh produce buyers in Europe.
You answer questions about weekly fruit and vegetable prices (ex-packaging station prices, national weekly averages)
for Belgium, the Netherlands, Germany, France, Spain and Italy.
Rules:
- Give prices in euro per kg with 2 decimals, and say which week the price is for.
- Only use numbers that come from your tools. Never guess or invent a price.
- If you can't find the data, say clearly that you don't have it.
- Keep the answer short: start with the direct answer in one sentence."""

SYSTEM_NO_TOOLS = SYSTEM.replace("Only use numbers that come from your tools.", "Only use numbers you are sure about.")

client = None


def get_client():
    global client
    if client is None:
        client = OpenAI()
    return client


def run_tool(name, args):
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except Exception as e:  # a bad argument should not crash the whole run
        return {"error": f"{type(e).__name__}: {e}"}


def ask(question, use_tools=True):
    messages = [{"role": "system", "content": SYSTEM if use_tools else SYSTEM_NO_TOOLS},
                {"role": "user", "content": question}]
    calls = []
    usage = {"input": 0, "output": 0}
    for _ in range(MAX_ROUNDS):
        kwargs = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": MAX_TOKENS}
        if use_tools:
            kwargs["tools"] = SCHEMAS
        resp = get_client().chat.completions.create(**kwargs)
        usage["input"] += resp.usage.prompt_tokens
        usage["output"] += resp.usage.completion_tokens
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "tool_calls": calls, "usage": usage}
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = run_tool(tc.function.name, args)
            calls.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return {"answer": "(no final answer after 5 tool rounds)", "tool_calls": calls, "usage": usage}


if __name__ == "__main__":
    q = "What is the current reported average price of tomatoes in Belgium?"
    for t in (True, False):
        r = ask(q, use_tools=t)
        print("tools" if t else "no tools", "->", r["answer"], r["tool_calls"])
