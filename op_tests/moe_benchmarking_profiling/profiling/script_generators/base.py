#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base class for script generators.

Defines the interface that all operator-specific script generators must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Any


class ScriptConfig(ABC):
    """
    Base class for script configuration.
    
    Subclasses should be dataclasses with operator-specific configuration fields.
    """
    
    @property
    @abstractmethod
    def unique_id(self) -> str:
        """Return a unique identifier for this configuration."""
        pass
    
    @property
    def script_name(self) -> str:
        """Return the filename for the generated script."""
        return f"{self.unique_id}.py"
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> 'ScriptConfig':
        """Create configuration from dictionary."""
        pass


class ScriptGenerator(ABC):
    """
    Abstract base class for script generators.
    
    Each operator (MOE, GEMM, Attention, etc.) should implement its own
    generator that creates Python scripts for profiling.
    """
    
    @abstractmethod
    def generate(self, config: ScriptConfig) -> str:
        """
        Generate Python script code for the given configuration.
        
        Args:
            config: Configuration for the script
            
        Returns:
            Complete Python script as string
        """
        pass
    
    def write_script(
        self,
        config: ScriptConfig,
        output_dir: Path,
        script_name: Optional[str] = None,
    ) -> Path:
        """
        Generate and write script to file.
        
        Args:
            config: Configuration for the script
            output_dir: Directory to write script to
            script_name: Optional custom script name (default: config.script_name)
            
        Returns:
            Path to written script
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = output_dir / (script_name or config.script_name)
        code = self.generate(config)
        script_path.write_text(code)
        script_path.chmod(0o755)
        
        return script_path
    
    def generate_all(
        self,
        configs: List[ScriptConfig],
        output_dir: Path,
    ) -> List[Path]:
        """
        Generate scripts for all configurations.
        
        Args:
            configs: List of configurations
            output_dir: Directory to write scripts to
            
        Returns:
            List of paths to written scripts
        """
        return [self.write_script(config, output_dir) for config in configs]
    
    @classmethod
    @abstractmethod
    def load_configs(cls, csv_file: Path, **kwargs) -> List[ScriptConfig]:
        """
        Load configurations from a CSV file.
        
        Args:
            csv_file: Path to CSV file
            **kwargs: Additional options
            
        Returns:
            List of configurations
        """
        pass
