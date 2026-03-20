from handlers.commands import CommandHandlers


class DummyBackend:
    async def get_items(self):
        return [
            {"type": "lab", "title": "Lab 01 - Intro"},
            {"type": "task", "title": "Task 1"},
        ]

    async def list_labs(self):
        return [{"id": "lab-01", "title": "Lab 01 - Intro"}]

    async def get_pass_rates(self, lab: str):
        if lab == "lab-01":
            return [{"task": "Repository Setup", "avg_score": 91.2, "attempts": 12}]
        return []


class DummyRouter:
    async def route(self, text: str) -> str:
        return f"routed: {text}"


async def test_help_lists_commands():
    handlers = CommandHandlers(backend=DummyBackend(), intent_router=DummyRouter())

    response = await handlers.handle_text("/help")

    assert "/start" in response
    assert "/scores <lab>" in response


async def test_start_returns_welcome_message():
    handlers = CommandHandlers(backend=DummyBackend(), intent_router=DummyRouter())

    response = await handlers.handle_text("/start")

    assert "Welcome to LMS Insight Bot" in response


async def test_labs_returns_backend_data():
    handlers = CommandHandlers(backend=DummyBackend(), intent_router=DummyRouter())

    response = await handlers.handle_text("/labs")

    assert "Available labs:" in response
    assert "lab-01 - Lab 01 - Intro" in response


async def test_scores_requires_argument():
    handlers = CommandHandlers(backend=DummyBackend(), intent_router=DummyRouter())

    response = await handlers.handle_text("/scores")

    assert "Usage: /scores <lab>" in response


async def test_natural_language_goes_through_router():
    handlers = CommandHandlers(backend=DummyBackend(), intent_router=DummyRouter())

    response = await handlers.handle_text("what labs are available?")

    assert response == "routed: what labs are available?"
