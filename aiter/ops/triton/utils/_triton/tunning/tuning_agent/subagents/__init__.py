"""Subagent package for the agentic Triton kernel tuning pipeline.

Each subagent encapsulates one discrete phase of the tuning pipeline (e.g.
environment setup, kernel discovery, benchmarking).  All subagents derive from
:class:`~tuning_agent.subagents.base.BaseSubagent` and communicate results via
:class:`~tuning_agent.subagents.base.SubagentResult`.
"""
from .base import BaseSubagent, SubagentResult, SubagentError
from .setup_agent import SetupAgent
from .discovery_agent import DiscoveryAgent
from .script_creator_agent import ScriptCreatorAgent
from .baseline_agent import BaselineAgent
from .tuning_agent import TuningAgent
from .pattern_analyzer_agent import PatternAnalyzerAgent
from .config_generator_agent import ConfigGeneratorAgent
from .validation_agent import ValidationAgent
from .regression_fixer_agent import RegressionFixerAgent

__all__ = [
    "BaseSubagent", "SubagentResult", "SubagentError",
    "SetupAgent", "DiscoveryAgent", "ScriptCreatorAgent",
    "BaselineAgent", "TuningAgent", "PatternAnalyzerAgent",
    "ConfigGeneratorAgent", "ValidationAgent", "RegressionFixerAgent",
]
