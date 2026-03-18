"""
vr_agent/executor.py
--------------------
DirectMCPExecutor: dynamically loads mcp_server.py and dispatches tool calls.
"""

import importlib.util
import sys


class DirectMCPExecutor:
    """
    Loads mcp_server.py at runtime via importlib so the agent can call
    any function in that module without a static import.
    """

    def __init__(self):
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        if not spec:
            raise ImportError("Could not find mcp_server.py in the current directory.")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.module
        spec.loader.exec_module(self.module)

    def call(self, tool: str, **kwargs):
        """Call a function in mcp_server by name, passing keyword arguments."""
        func = getattr(self.module, tool)
        return func(**kwargs)
