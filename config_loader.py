"""
Configuration loader for Exam-Helper application.
Handles loading and managing configuration from YAML file.
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """Handles loading and accessing configuration from YAML file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get_api_keys(self, google_key: Optional[str] = None) -> Dict[str, str]:
        """
        Get API keys configuration.
        
        Args:
            google_key (str, optional): User-provided Google API key
            
        Returns:
            Dict[str, str]: Dictionary of API keys
        """
        api_keys = self._config.get("api_keys", {}).copy()
        if google_key:
            # Override the default Google API key with user-provided one
            api_keys["google"] = google_key
            api_keys["google_vision"] = google_key
        return api_keys
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self._config.get("app_config", {})
    
    def get_models(self) -> list:
        """Get available models list."""
        return self._config.get("models", [])
    
    def get_document_indexes(self) -> list:
        """Get available document indexes."""
        return self._config.get("document_indexes", [])
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get("ui_config", {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# Global configuration instance
config = ConfigLoader()