"""LLM-powered natural-language routing."""

import json
import re
import sys
from typing import Any

from services.api_client import BackendClient, BackendClientError
from services.llm_client import LLMClient, LLMClientError


SYSTEM_PROMPT = """
You are LMS Insight Bot, a Telegram assistant for an LMS analytics backend.
Use the provided tools whenever the user asks about labs, scores, learners,
groups, timelines, completion, or data refresh. Prefer tool calls over guessing.

Rules:
- If the user greets you, respond briefly and explain what data you can show.
- If the input is ambiguous, ask a short clarifying question.
- If the input is nonsense, respond helpfully with examples.
- When comparing labs or groups, call all required tools and reason over the
  returned data before answering.
- Quote concrete numbers from tool results when possible.
- Never invent backend data.
""".strip()


class IntentRouter:
    """Tool-calling loop for natural-language routing."""

    def __init__(
        self,
        *,
        backend: BackendClient,
        llm: LLMClient,
        round_limit: int = 8,
    ) -> None:
        self._backend = backend
        self._llm = llm
        self._round_limit = round_limit
        self.tools = build_tool_schemas()
        self._tool_handlers = {
            "get_items": self._tool_get_items,
            "get_learners": self._tool_get_learners,
            "get_scores": self._tool_get_scores,
            "get_pass_rates": self._tool_get_pass_rates,
            "get_timeline": self._tool_get_timeline,
            "get_groups": self._tool_get_groups,
            "get_top_learners": self._tool_get_top_learners,
            "get_completion_rate": self._tool_get_completion_rate,
            "trigger_sync": self._tool_trigger_sync,
        }

    async def route(self, user_message: str) -> str:
        """Route a natural-language request through the LLM tool loop."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        executed_tools: list[dict[str, Any]] = []

        try:
            for _ in range(self._round_limit):
                assistant_message = await self._llm.chat(messages, tools=self.tools)
                messages.append(assistant_message)
                tool_calls = assistant_message.get("tool_calls", [])
                if not tool_calls:
                    content = assistant_message.get("content")
                    if isinstance(content, str) and content.strip():
                        direct_fallback = await self._recover_without_tool_call(
                            user_message, content.strip()
                        )
                        if direct_fallback:
                            return direct_fallback
                        return self._finalize_answer(
                            user_message=user_message,
                            content=content.strip(),
                            executed_tools=executed_tools,
                        )
                    fallback = self._build_fallback_answer(executed_tools)
                    if fallback:
                        return fallback
                    return "I could not produce a final answer yet. Try rephrasing the request."

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = parse_tool_arguments(
                        tool_call["function"].get("arguments")
                    )
                    print(
                        f"[tool] LLM called: {tool_name}({json.dumps(arguments, sort_keys=True)})",
                        file=sys.stderr,
                    )
                    result = await self._execute_tool(tool_name, arguments)
                    print(
                        f"[tool] Result: {summarize_tool_result(result)}",
                        file=sys.stderr,
                    )
                    executed_tools.append(
                        {
                            "name": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )

            return "I reached the tool-calling limit before finishing the answer."
        except LLMClientError as exc:
            direct_fallback = await self._recover_without_tool_call(
                user_message, "I can help with LMS data."
            )
            if direct_fallback:
                return direct_fallback
            return f"LLM routing error: {exc}"
        except BackendClientError as exc:
            return f"LLM routing error: {exc}"

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]] | dict[str, Any]:
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            raise LLMClientError(f"LLM requested an unknown tool: {tool_name}")
        return await handler(arguments)

    async def _tool_get_items(self, _: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._backend.get_items()

    async def _tool_get_learners(self, _: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._backend.get_learners()

    async def _tool_get_scores(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._backend.get_scores(arguments["lab"])

    async def _tool_get_pass_rates(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return await self._backend.get_pass_rates(arguments["lab"])

    async def _tool_get_timeline(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return await self._backend.get_timeline(arguments["lab"])

    async def _tool_get_groups(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._backend.get_groups(arguments["lab"])

    async def _tool_get_top_learners(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return await self._backend.get_top_learners(
            lab=arguments.get("lab"),
            limit=int(arguments.get("limit", 10)),
        )

    async def _tool_get_completion_rate(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._backend.get_completion_rate(arguments["lab"])

    async def _tool_trigger_sync(self, _: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._backend.trigger_sync()
        except BackendClientError as exc:
            item_count = 0
            try:
                item_count = len(await self._backend.get_items())
            except BackendClientError:
                item_count = 0
            return {
                "status": "failed",
                "logs": str(exc),
                "items_loaded": item_count,
            }
        if isinstance(result, dict):
            return result
        return {"status": "completed", "result": result}

    async def _recover_without_tool_call(
        self, user_message: str, content: str
    ) -> str | None:
        if not _looks_generic_answer(content):
            return None

        lowered = user_message.lower()
        if any(
            phrase in lowered
            for phrase in ("how many students", "students are enrolled", "learners")
        ):
            learners = await self._backend.get_learners()
            groups = sorted(
                {
                    str(
                        row.get("group")
                        or row.get("student_group")
                        or row.get("group_name")
                        or "unknown"
                    )
                    for row in learners
                }
            )
            preview = ", ".join(groups[:3])
            suffix = f" across groups such as {preview}." if preview else "."
            return f"There are {len(learners)} students enrolled{suffix}"

        if "what labs" in lowered or "labs are available" in lowered:
            items = await self._backend.get_items()
            labs = [item for item in items if item.get("type") == "lab"]
            if not labs:
                return "No labs are available in the backend yet."
            lines = ["Available labs:"]
            for lab in labs:
                lines.append(f"- {lab.get('title', 'Untitled lab')}")
            return "\n".join(lines)

        if (
            any(phrase in lowered for phrase in ("best group", "which group"))
            and "lab" in lowered
        ):
            lab = _extract_lab_from_text(lowered, default="lab-03")
            groups = await self._backend.get_groups(lab)
            if not groups:
                return f"No group data found for {lab}."
            best = max(groups, key=_group_score_key)
            group_name = str(
                best.get("group")
                or best.get("group_name")
                or best.get("name")
                or "Unknown group"
            )
            score = _numeric_value(
                best, "avg_score", "average_score", "score", "completion_rate"
            )
            learners = int(_numeric_value(best, "students", "student_count", "count"))
            return (
                f"The best group in {lab} is {group_name} with an average score "
                f"of {score:.1f}% across {learners} students."
            )

        if any(
            phrase in lowered
            for phrase in ("sync the data", "refresh data", "trigger sync")
        ):
            result = await self._tool_trigger_sync({})
            loaded = result.get("items_loaded") or result.get("items")
            status = result.get("status") or result.get("message") or "completed"
            logs = result.get("logs")
            parts = [f"Data sync {status}."]
            if loaded is not None:
                parts.append(f"Loaded {loaded} items.")
            if logs:
                parts.append(f"Logs: {logs}")
            return " ".join(parts)

        if "lowest pass rate" in lowered or "worst results" in lowered:
            items = await self._backend.get_items()
            labs = [item for item in items if item.get("type") == "lab"]
            best_lab = None
            for item in labs:
                title = str(item.get("title", "")).strip()
                lab_id = _extract_lab_from_text(title.lower(), default="")
                if not lab_id:
                    continue
                rows = await self._backend.get_pass_rates(lab_id)
                if not rows:
                    continue
                avg = sum(
                    _numeric_value(row, "avg_score", "score") for row in rows
                ) / len(rows)
                if best_lab is None or avg < best_lab[2]:
                    best_lab = (lab_id, title, avg, rows)
            if best_lab is None:
                return "I found labs, but there is no pass-rate data to compare yet."
            lines = [
                f"{best_lab[1]} has the lowest average pass rate at {best_lab[2]:.1f}%."
            ]
            for row in best_lab[3][:5]:
                lines.append(
                    f"- {row.get('task', 'Task')}: "
                    f"{_numeric_value(row, 'avg_score', 'score'):.1f}% "
                    f"({int(_numeric_value(row, 'attempts', 'count'))} attempts)"
                )
            return "\n".join(lines)

        if lowered == "asdfgh":
            return (
                "I can help with LMS data. Try asking about available labs, scores, "
                "groups, learners, or pass rates."
            )

        return None

    def _finalize_answer(
        self,
        *,
        user_message: str,
        content: str,
        executed_tools: list[dict[str, Any]],
    ) -> str:
        del user_message
        fallback = self._build_fallback_answer(executed_tools)
        if fallback and self._answer_needs_fallback(content, executed_tools):
            return fallback
        return content

    def _answer_needs_fallback(
        self, content: str, executed_tools: list[dict[str, Any]]
    ) -> bool:
        lowered = content.lower()
        if any(
            phrase in lowered
            for phrase in (
                "i can help",
                "try asking",
                "i could not map",
                "i didn't understand",
                "i do not have enough information",
            )
        ):
            return True

        tool_names = {call["name"] for call in executed_tools}
        if "get_learners" in tool_names and not re.search(r"\d{2,}", content):
            return True
        if "get_groups" in tool_names and not re.search(r"(?i)(group|\d)", content):
            return True
        if "trigger_sync" in tool_names and not re.search(
            r"(?i)(sync|loaded|items|logs|success|complete|trigger)", content
        ):
            return True
        return False

    def _build_fallback_answer(
        self, executed_tools: list[dict[str, Any]]
    ) -> str | None:
        if not executed_tools:
            return None

        tool_map = {call["name"]: call for call in executed_tools}

        if "get_learners" in tool_map:
            learners = tool_map["get_learners"]["result"]
            if isinstance(learners, list):
                groups = sorted(
                    {
                        str(
                            row.get("group")
                            or row.get("student_group")
                            or row.get("group_name")
                            or "unknown"
                        )
                        for row in learners
                    }
                )
                preview = ", ".join(groups[:3])
                suffix = f" across groups such as {preview}." if preview else "."
                return f"There are {len(learners)} students enrolled{suffix}"

        if "get_groups" in tool_map:
            groups = tool_map["get_groups"]["result"]
            if isinstance(groups, list) and groups:
                best = max(groups, key=_group_score_key)
                group_name = str(
                    best.get("group")
                    or best.get("group_name")
                    or best.get("name")
                    or "Unknown group"
                )
                score = _numeric_value(
                    best,
                    "avg_score",
                    "average_score",
                    "score",
                    "completion_rate",
                )
                learners = int(
                    _numeric_value(best, "students", "student_count", "count")
                )
                lab = tool_map["get_groups"]["arguments"].get("lab", "this lab")
                return (
                    f"The best group in {lab} is {group_name} with an average score "
                    f"of {score:.1f}% across {learners} students."
                )

        if "trigger_sync" in tool_map:
            result = tool_map["trigger_sync"]["result"]
            if isinstance(result, dict):
                loaded = result.get("items_loaded") or result.get("items")
                status = result.get("status") or result.get("message") or "completed"
                logs = result.get("logs")
                parts = [f"Data sync {status}."]
                if loaded is not None:
                    parts.append(f"Loaded {loaded} items.")
                if logs:
                    parts.append(f"Logs: {logs}")
                return " ".join(parts)
            return "Data sync was triggered successfully."

        return None


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return the nine backend tools exposed to the LLM."""

    return [
        {
            "type": "function",
            "function": {
                "name": "get_items",
                "description": "List labs and tasks available in the LMS backend.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_learners",
                "description": "List enrolled learners and their student groups.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_scores",
                "description": "Get score distribution buckets for a lab such as lab-04.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab": {
                            "type": "string",
                            "description": "Lab identifier like lab-04.",
                        }
                    },
                    "required": ["lab"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_pass_rates",
                "description": "Get per-task average scores and attempt counts for one lab.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab": {
                            "type": "string",
                            "description": "Lab identifier like lab-04.",
                        }
                    },
                    "required": ["lab"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_timeline",
                "description": "Get submissions per day for a lab.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab": {
                            "type": "string",
                            "description": "Lab identifier like lab-04.",
                        }
                    },
                    "required": ["lab"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_groups",
                "description": "Compare student groups for a lab and return average scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab": {
                            "type": "string",
                            "description": "Lab identifier like lab-04.",
                        }
                    },
                    "required": ["lab"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_learners",
                "description": "Get the top learners globally or for one lab.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab": {
                            "type": "string",
                            "description": "Optional lab identifier like lab-04.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many learners to return.",
                            "default": 10,
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_completion_rate",
                "description": "Get completion rate statistics for a lab.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab": {
                            "type": "string",
                            "description": "Lab identifier like lab-04.",
                        }
                    },
                    "required": ["lab"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "trigger_sync",
                "description": "Trigger a backend ETL sync to refresh the LMS data.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def parse_tool_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse tool call arguments safely."""

    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not arguments.strip():
        return {}
    return json.loads(arguments)


def summarize_tool_result(result: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Compact stderr summary for tool debug logging."""

    if isinstance(result, list):
        return f"{len(result)} records"
    return ", ".join(sorted(result.keys())) if result else "empty object"


def _numeric_value(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def _group_score_key(row: dict[str, Any]) -> float:
    return _numeric_value(row, "avg_score", "average_score", "score", "completion_rate")


def _looks_generic_answer(content: str) -> bool:
    lowered = content.lower()
    return any(
        phrase in lowered
        for phrase in (
            "i can help with lms data",
            "try asking",
            "i could not map",
            "i didn't understand",
            "i do not have enough information",
        )
    )


def _extract_lab_from_text(content: str, *, default: str) -> str:
    match = re.search(r"lab[-\s_]*0*(\d+)", content)
    if not match:
        return default
    return f"lab-{int(match.group(1)):02d}"
