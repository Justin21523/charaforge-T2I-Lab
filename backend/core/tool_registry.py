# backend/core/tool_registry.py
import yaml
import importlib
import inspect
from typing import Dict, Any, Callable
from pathlib import Path


class ToolRegistry:
    def __init__(self, config_path: str = "configs/agent.yaml"):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.config_path = config_path
        self.reload_tools()

    def reload_tools(self):
        """Reload tools from YAML config"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.tools.clear()
            for tool_config in config.get("tools", []):
                self._register_tool(tool_config)
        except Exception as e:
            print(f"[ToolRegistry] Failed to load config: {e}")

    def _register_tool(self, tool_config: Dict[str, Any]):
        """Register a single tool from config"""
        try:
            name = tool_config["name"]
            function_path = tool_config["function"]

            # Import and get function
            module_path, func_name = function_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)

            # Validate function signature
            sig = inspect.signature(func)

            self.tools[name] = {
                "function": func,
                "description": tool_config.get("description", ""),
                "parameters": tool_config.get("parameters", {}),
                "signature": sig,
                "safety_level": tool_config.get("safety_level", "safe"),
            }
            print(f"[ToolRegistry] Registered tool: {name}")

        except Exception as e:
            print(
                f"[ToolRegistry] Failed to register {tool_config.get('name', 'unknown')}: {e}"
            )

    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get tool name -> description mapping for LLM context"""
        return {name: info["description"] for name, info in self.tools.items()}
