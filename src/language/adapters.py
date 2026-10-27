"""Which LoRA adapter serves which task of the local model (``adapters.toml``).

The local model is one base model and a LoRA adapter per group of tasks, instead of one model
fine-tuned on everything. An adapter is trained on its own tasks' rows (``notebooks/
train_multigpu_kaggle.ipynb``: ``ADAPTER``, ``TASKS``), can be updated from its previous version
with only the new rows (``INIT_FROM``), and is exported as a small LoRA GGUF that the server applies
to the base it already holds. Changing what one task does retrains one adapter; the others, the base
and their cached evaluations stay as they are.

    base = "Qwen3.5-4B-Base-Q8_0"      # the model the server serves the adapters on
    [adapters]                          # adapter (its GGUF file name) -> the tasks it was trained on
    nts-intents = ["compile"]
    nts-policy = ["policy"]
    [domains.iitx]                      # a domain's own adapters, task -> adapter; the rest as above
    policy = "iitx-policy"

Tasks are the ``task`` of the training rows (``training.distill``): ``compile`` (reading requests),
``policy``, ``reply`` and ``explain``; and ``lookup`` (``training.lookup_data``: which timetable lookup a
question asks for, ``agents.tool_agent``). ``NTS_ADAPTERS`` names the file (the web app and
``--adapters`` read it) and ``NTS_DOMAIN`` the domain. Without a file, every task goes to the one
model the server serves, as before.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError

TASKS = ("compile", "policy", "reply", "explain", "lookup", "memory")  # lookup: agents.tool_agent (timetable
# questions); memory: agents.memory (a follow-up made to stand on its own)


@dataclass
class Registry:
    base: str | None = None
    adapters: dict[str, list[str]] = field(default_factory=dict)
    domains: dict[str, dict[str, str]] = field(default_factory=dict)
    # the intent vocabulary version an adapter was trained on (``intents.SCHEMA_VERSION``), when not the current
    schemas: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> Registry:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        reg = cls(base=data.get("base"), adapters={k: list(v) for k, v in data.get("adapters", {}).items()},
                  domains={k: dict(v) for k, v in data.get("domains", {}).items()},
                  schemas={k: int(v) for k, v in data.get("schemas", {}).items()})
        for name, tasks in reg.adapters.items():
            if unknown := set(tasks) - set(TASKS):
                raise ValueError(f"{path}: adapter {name!r} has unknown tasks {sorted(unknown)}; tasks are {TASKS}")
        for task in TASKS:
            if len(serving := [a for a, t in reg.adapters.items() if task in t]) > 1:
                raise ValueError(f"{path}: task {task!r} is served by more than one adapter: {serving}")
        for domain, tasks in reg.domains.items():
            if unknown := set(tasks) - set(TASKS):
                raise ValueError(f"{path}: domain {domain!r} has unknown tasks {sorted(unknown)}")
        return reg

    def adapter_for(self, task: str, domain: str | None = None) -> str | None:
        """The adapter for ``task``: the domain's own if it has one, else the shared one, else none (the
        served model itself)."""
        if domain is not None and domain not in self.domains:
            raise ValueError(f"unknown domain {domain!r}; adapters.toml has {sorted(self.domains)}")
        if domain and task in self.domains[domain]:
            return self.domains[domain][task]
        return next((a for a, tasks in self.adapters.items() if task in tasks), None)

    def describe(self, domain: str | None = None) -> str:
        return ", ".join(f"{t}: {self.adapter_for(t, domain) or self.base or 'served model'}" for t in TASKS)


def registry(path: Path | str | None = None) -> Registry | None:
    """The registry at ``path``, else at ``NTS_ADAPTERS``, else none."""
    path = path or os.environ.get("NTS_ADAPTERS")
    return Registry.load(path) if path else None


class TaskClients:
    """A client per task, made on first use: the web app's model link holds one of these, and each
    component takes its task's client (``clients.get("policy")``)."""

    def __init__(self, make: Callable[[str], object]) -> None:
        self.make = make
        self._made: dict[str, object] = {}

    def get(self, task: str):
        if task not in self._made:
            self._made[task] = self.make(task)
        return self._made[task]


def local_client(task: str, *, base_url: str, adapter: str | None = None, server_model: str | None = None,
                 reg: Registry | None = None, domain: str | None = None, **kwargs):
    """A ``LocalClient`` for ``task``: on ``adapter`` if given, else on the registry's adapter for the task,
    else on the served model (``server_model`` / ``NTS_LOCAL_MODEL``)."""
    from .local import LocalClient

    domain = domain or os.environ.get("NTS_DOMAIN") or None
    if adapter is None and reg is not None:
        adapter = reg.adapter_for(task, domain)
    if server_model is None and reg is not None:  # the adapters' base, or the base itself for a task without one
        server_model = reg.base
    schema = kwargs.pop("schema", None)
    try:
        client = LocalClient(base_url=base_url, server_model=server_model, adapter=adapter, **kwargs)
    except LLMError as e:
        raise LLMError(f"{task}: {e}") from e
    # the vocabulary it reads in: given, else the registry's for the adapter, else the current one
    client.schema = schema or (reg.schemas.get(adapter or server_model or "") if reg is not None else None)
    return client


def add_args(ap) -> None:
    """``--adapter`` and ``--adapters`` for a command that runs the local model."""
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter to run the task on (its GGUF name; see language.adapters)")
    ap.add_argument("--adapters", type=Path, default=None,
                    help="adapter registry (default NTS_ADAPTERS): each task on its adapter")
    ap.add_argument("--schema", type=int, default=None,
                    help="the intent vocabulary version the model was trained on (default: the registry's, "
                         "else the current); an older one is asked with its prompt and its answer migrated")
    ap.add_argument("--zero-shot", action="store_true",
                    help="the model was not fine-tuned on reading requests (a base model): give it the whole "
                         "vocabulary in the prompt, as Gemini is")


def task_clients(*, base_url: str, reg: Registry | None = None, domain: str | None = None,
                 **kwargs) -> TaskClients:
    """Clients for every task: one shared client without a registry, else one per adapter."""
    if reg is None:
        from .local import LocalClient

        one = LocalClient(base_url=base_url, **kwargs)
        return TaskClients(lambda task: one)
    by_adapter: dict[str | None, object] = {}

    def make(task: str):
        name = reg.adapter_for(task, domain or os.environ.get("NTS_DOMAIN") or None)
        if name not in by_adapter:
            by_adapter[name] = local_client(task, base_url=base_url, adapter=name, reg=reg, domain=domain, **kwargs)
        return by_adapter[name]

    return TaskClients(make)
