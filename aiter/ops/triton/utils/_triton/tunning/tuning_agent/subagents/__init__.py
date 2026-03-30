"""Subagent package for the agentic Triton kernel tuning pipeline.

Each subagent encapsulates one discrete phase of the tuning pipeline (e.g.
environment setup, kernel discovery, benchmarking).  All subagents derive from
:class:`~tuning_agent.subagents.base.BaseSubagent` and communicate results via
:class:`~tuning_agent.subagents.base.SubagentResult`.
"""
