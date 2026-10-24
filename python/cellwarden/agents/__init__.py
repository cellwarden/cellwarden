"""LLM agents (docs/DESIGN.md section 4, docs/agents-spec.md).

* :mod:`cellwarden.agents.base`: the agent base class (prompt file + tools + a
  capped tool-use loop), budget guard and replay hooks.
* :mod:`cellwarden.agents.qc_agent`: the G1 QC worker and the proposal logic.
* :mod:`cellwarden.agents.reviewer`: the reviewer agent used by every gate.
* ``prompts/<role>.md``: the system prompts.

Other roles (context, annotation, interpretation) are added by their own
modules in later build steps.
"""

from .base import Agent, AgentError, AgentResult, Tool, load_prompt, wrap_data

__all__ = ["Agent", "AgentError", "AgentResult", "Tool", "load_prompt", "wrap_data"]
