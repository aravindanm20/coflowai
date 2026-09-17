"""Durable persistence adapters.

Import the backend you need — nothing here is loaded by ``import coflowai``:

    from coflowai.persistence.sqlite import SqliteEventStore, SqliteStateStore

* ``sqlite``   — zero extra dependencies (stdlib), durable, single node
* ``postgres`` — multi-worker, atomic lock leases (requires ``asyncpg``)
* ``redis``    — multi-worker, fenced locks + queues (requires ``redis>=5``)
"""

__all__: list[str] = []
