"""Provider adapters.

Adapters are imported explicitly and lazily — importing :mod:`coflowai` never
imports a provider SDK, which keeps the core dependency-free (Rule 3).

    from coflowai.providers.openai import OpenAIProvider, OPENAI_CAPABILITIES

    app.models.register("smart-model",
                        OpenAIProvider(model="gpt-4o", api_key=...),
                        capabilities=OPENAI_CAPABILITIES)

``template.py`` documents the full contract for writing a new adapter.
"""

from .http import HttpTransport, HttpxTransport, MockTransport

__all__ = ["HttpTransport", "HttpxTransport", "MockTransport"]
