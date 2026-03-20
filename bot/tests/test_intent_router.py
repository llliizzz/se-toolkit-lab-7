from services.intent_router import IntentRouter


class DummyBackend:
    async def get_items(self):
        return [{"type": "lab", "title": "Lab 03 - Backend API"}]

    async def get_learners(self):
        return [
            {"name": "Alice", "group": "Group A"},
            {"name": "Bob", "group": "Group B"},
            {"name": "Carol", "group": "Group A"},
        ]

    async def get_scores(self, lab: str):
        return [{"bucket": "80-100", "count": 2, "lab": lab}]

    async def get_pass_rates(self, lab: str):
        return [{"task": "Backend API", "avg_score": 71.5, "attempts": 4, "lab": lab}]

    async def get_timeline(self, lab: str):
        return [{"date": "2026-03-20", "count": 2, "lab": lab}]

    async def get_groups(self, lab: str):
        return [
            {"group": "Group A", "avg_score": 74.0, "student_count": 12, "lab": lab},
            {"group": "Group B", "avg_score": 88.0, "student_count": 10, "lab": lab},
        ]

    async def get_top_learners(self, lab=None, limit=10):
        return [{"name": "Alice", "score": 98.0, "lab": lab, "limit": limit}]

    async def get_completion_rate(self, lab: str):
        return {"completion_rate": 83.0, "lab": lab}

    async def trigger_sync(self):
        return {"status": "completed", "items_loaded": 14}


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    async def chat(self, messages, *, tools=None, tool_choice="auto"):
        del messages, tools, tool_choice
        return self._responses.pop(0)


async def test_router_falls_back_to_learner_count_summary():
    router = IntentRouter(
        backend=DummyBackend(),
        llm=ScriptedLLM(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_learners",
                            "function": {"name": "get_learners", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "I can help with LMS data.",
                },
            ]
        ),
    )

    response = await router.route("how many students are enrolled?")

    assert "3 students enrolled" in response


async def test_router_falls_back_to_group_comparison_summary():
    router = IntentRouter(
        backend=DummyBackend(),
        llm=ScriptedLLM(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_groups",
                            "function": {
                                "name": "get_groups",
                                "arguments": '{"lab":"lab-03"}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Here is the comparison.",
                },
            ]
        ),
    )

    response = await router.route("which group is doing best in lab 3?")

    assert "Group B" in response
    assert "88.0%" in response


async def test_router_falls_back_to_sync_summary():
    router = IntentRouter(
        backend=DummyBackend(),
        llm=ScriptedLLM(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_sync",
                            "function": {"name": "trigger_sync", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Done.",
                },
            ]
        ),
    )

    response = await router.route("sync the data")

    assert "sync" in response.lower()
    assert "14 items" in response


async def test_router_recovers_learner_count_when_llm_skips_tools():
    router = IntentRouter(
        backend=DummyBackend(),
        llm=ScriptedLLM(
            [{"role": "assistant", "content": "I can help with LMS data."}]
        ),
    )

    response = await router.route("how many students are enrolled?")

    assert "3 students enrolled" in response


async def test_router_recovers_group_summary_when_llm_skips_tools():
    router = IntentRouter(
        backend=DummyBackend(),
        llm=ScriptedLLM(
            [{"role": "assistant", "content": "I can help with LMS data."}]
        ),
    )

    response = await router.route("which group is doing best in lab 3?")

    assert "Group B" in response
    assert "88.0%" in response


async def test_router_recovers_sync_summary_when_llm_skips_tools():
    router = IntentRouter(
        backend=DummyBackend(),
        llm=ScriptedLLM(
            [{"role": "assistant", "content": "I can help with LMS data."}]
        ),
    )

    response = await router.route("sync the data")

    assert "sync" in response.lower()
    assert "14 items" in response
