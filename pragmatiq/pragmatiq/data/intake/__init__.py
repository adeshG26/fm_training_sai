"""Real-data intake: client tables in, pragmatiq's data contract out.

Replaces the synthetic generator as the source of ``events.parquet`` and
``profiles.parquet`` (plus ``labels/*.parquet`` via :mod:`.labels`). One YAML
mapping per client describes the tables; nothing else here knows a client's
column names. Public entry points are the ``intake_*`` / ``prepare`` functions
in :mod:`pragmatiq.api`.
"""
