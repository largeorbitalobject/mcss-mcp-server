#!/usr/bin/env python
"""
MCSS MCP Server - Main Entry Point
"""
import os
import sys
from mcss_mcp.server import mcp

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
