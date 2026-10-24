"""LLM adapter, budget ledger and agent call log (docs/agents-spec.md section 3).

::

    from cellwarden.llm import get_provider, Ledger, AgentLog
    provider = get_provider(cfg["llm"], role="qc_worker")
    resp = provider.complete(system=..., messages=[...], tools=[...], max_tokens=1024)

See :mod:`cellwarden.llm.provider` for the interface and message format,
:mod:`cellwarden.llm.ledger` for the budget guard and the replayable call log.
"""

from .ledger import AgentLog, BudgetExceeded, Ledger, load_prices, price_for, sha256_text
from .provider import (
    JSON_TOOL,
    AUTO_ORDER,
    PRESETS,
    PROVIDERS,
    apply_preset,
    auto_provider,
    Provider,
    ProviderError,
    Response,
    ToolCall,
    Usage,
    get_provider,
    role_settings,
)

__all__ = [
    "AgentLog",
    "BudgetExceeded",
    "Ledger",
    "load_prices",
    "price_for",
    "sha256_text",
    "JSON_TOOL",
    "AUTO_ORDER",
    "PRESETS",
    "PROVIDERS",
    "apply_preset",
    "auto_provider",
    "Provider",
    "ProviderError",
    "Response",
    "ToolCall",
    "Usage",
    "get_provider",
    "role_settings",
]
