#!/usr/bin/env python3
"""
MCP Server for MCSS API
This server provides tools to interact with the MCSS (Minecraft Server Software) API
through the Model Context Protocol (MCP).
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

# Add the packages directory to the Python path
packages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages")
sys.path.insert(0, packages_dir)

import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("mcss-control")

# Constants
DEFAULT_HOST = os.environ.get("MCSS_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("MCSS_PORT", "8821"))  # Default MCSS web panel port
API_KEY = os.environ.get("MCSS_API_KEY", "")  # Get API key from environment variable


class MCSSClient:
    """Client for interacting with the MCSS API."""
    
    def __init__(self, host: str = None, port: int = None, api_key: str = None):
        """Initialize the MCSS API client.
        
        Args:
            host: The hostname or IP address of the MCSS server
            port: The port of the MCSS web panel
            api_key: The API key for authentication
        """
        self.host = host or os.environ.get("MCSS_HOST", DEFAULT_HOST)
        self.port = port or int(os.environ.get("MCSS_PORT", DEFAULT_PORT))
        self.api_key = api_key or os.environ.get("MCSS_API_KEY", API_KEY)
        self.base_url = f"http://{self.host}:{self.port}/api/v2"
        self.headers = {"apiKey": self.api_key}
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request to the MCSS API.
        
        Args:
            endpoint: The API endpoint to request
            
        Returns:
            The JSON response from the API
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}{endpoint}", headers=self.headers)
            response.raise_for_status()
            
            # Handle empty responses
            if not response.text or response.text.isspace():
                return {}
                
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": response.text}
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the MCSS API.
        
        Args:
            endpoint: The API endpoint to request
            data: The JSON data to send
            
        Returns:
            The JSON response from the API
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{endpoint}", 
                json=data, 
                headers=self.headers
            )
            response.raise_for_status()
            
            # Handle empty responses
            if not response.text or response.text.isspace():
                return {}
                
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": response.text}

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request to the MCSS API.
        
        Args:
            endpoint: The API endpoint to request
            
        Returns:
            The JSON response from the API
        """
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{self.base_url}{endpoint}", headers=self.headers)
            response.raise_for_status()
            
            # Handle empty responses
            if not response.text or response.text.isspace():
                return {}
                
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": response.text}


# Initialize MCSS client with environment variables
mcss_client = MCSSClient(os.environ.get("MCSS_HOST", DEFAULT_HOST), int(os.environ.get("MCSS_PORT", DEFAULT_PORT)), os.environ.get("MCSS_API_KEY", API_KEY))


# MCP Tools for MCSS API
@mcp.tool()
async def get_servers() -> str:
    """Get a list of all Minecraft servers managed by MCSS.
    
    Returns:
        A formatted string containing information about all servers
    """
    try:
        response = await mcss_client.get("/servers")
        servers = response  # The response is already a list of servers
        
        if not servers:
            return "No servers found."
        
        result = []
        for server in servers:
            server_info = (
                f"Server ID: {server.get('serverId')}\n"
                f"Name: {server.get('name')}\n"
                f"Status: {server.get('status')}\n"
                f"Description: {server.get('description', 'No description')}\n"
                f"Type: {server.get('type')}\n"
            )
            result.append(server_info)
        
        return "\n---\n".join(result)
    except Exception as e:
        return f"Error fetching servers: {str(e)}"


@mcp.tool()
async def get_server_details(server_id: str) -> str:
    """Get detailed information about a specific Minecraft server.
    
    Args:
        server_id: The ID of the server to get details for
    
    Returns:
        A formatted string containing detailed information about the server
    """
    try:
        response = await mcss_client.get(f"/servers/{server_id}")
        server = response  # The response is the server details
        
        if not server:
            return "No server details found."
        
        details = [
            f"Server ID: {server.get('serverId', 'N/A')}",
            f"Name: {server.get('name', 'N/A')}",
            f"Status: {server.get('status', 'N/A')}",
            f"Type: {server.get('type', 'N/A')}",
            f"Path: {server.get('pathToFolder', 'N/A')}",
            f"Auto Start: {'Yes' if server.get('isSetToAutoStart', False) else 'No'}",
            f"Force Save on Stop: {'Yes' if server.get('forceSaveOnStop', False) else 'No'}",
            f"Java Memory: {server.get('javaAllocatedMemory', 'N/A')} MB"
        ]
        
        return "\n".join(details)
    except Exception as e:
        return f"Error getting server details: {str(e)}"


@mcp.tool()
async def execute_server_action(server_id: str, action: str) -> str:
    """Execute a power action on a Minecraft server.
    
    Args:
        server_id: The ID of the server to execute the action on
        action: The action to execute (start, stop, restart, kill)
    
    Returns:
        A message indicating the result of the action
    """
    action_map = {
        "start": 2,
        "stop": 1,
        "restart": 4,
        "kill": 3
    }
    
    if action.lower() not in action_map:
        return f"Invalid action: {action}. Valid actions are: start, stop, restart, kill."
    
    action_code = action_map[action.lower()]
    
    try:
        data = {
            "action": action_code,
            "serverIds": [server_id]
        }
        response = await mcss_client.post("/servers/execute/action", data)
        
        # If the response is empty or None, consider it a success
        # The MCSS API returns an empty response for successful actions
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully executed {action} on server {server_id}."
        
        # If we got a response, check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully executed {action} on server {server_id}."
        else:
            return f"Failed to execute {action} on server {server_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error executing server action: {str(e)}"


@mcp.tool()
async def execute_server_command(server_id: str, command: str) -> str:
    """Execute a Minecraft command on a server.
    
    Args:
        server_id: The ID of the server to execute the command on
        command: The Minecraft command to execute
    
    Returns:
        A message indicating the result of the command execution
    """
    
    try:
        data = {
            "serverIds": [server_id],
            "commands": [command]
        }
        response = await mcss_client.post("/servers/execute/commands", data)
        
        # If the response is empty or None, consider it a success
        # The MCSS API returns an empty response for successful commands
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully executed command '{command}' on server {server_id}."
        
        # If we got a response, check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully executed command '{command}' on server {server_id}."
        else:
            return f"Failed to execute command on server {server_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error executing server command: {str(e)}"


@mcp.tool()
async def get_server_console(server_id: str, amount_of_lines: int = 100, take_from_beginning: bool = False, reversed: bool = False) -> str:
    """Get the console output of a Minecraft server.
    
    Args:
        server_id: The ID of the server to get console output for
        amount_of_lines: Number of console lines to retrieve (-1 for all lines)
        take_from_beginning: If True, start from the oldest lines first
        reversed: If True, reverse the order of the lines
    
    Returns:
        The console output of the server
    """
    
    try:
        # Add query parameters for the console endpoint
        params = {
            "amountOfLines": amount_of_lines,
            "takeFromBeginning": str(take_from_beginning).lower(),
            "reversed": str(reversed).lower()
        }
        
        # Use httpx directly for this request to include query parameters
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{mcss_client.base_url}/servers/{server_id}/console", 
                headers=mcss_client.headers,
                params=params
            )
            response.raise_for_status()
            
            # Parse the response
            if not response.text or response.text.isspace():
                return "No console output available."
                
            try:
                console_lines = response.json()
                if not console_lines or len(console_lines) == 0:
                    return "No console output available."
                
                return "\n".join(console_lines)
            except json.JSONDecodeError:
                return f"Error parsing console output: {response.text}"
    except Exception as e:
        return f"Error getting server console: {str(e)}"


@mcp.tool()
async def get_server_backups(server_id: str) -> str:
    """Get a list of backups for a Minecraft server.
    
    Args:
        server_id: The ID of the server to get backups for
    
    Returns:
        A formatted string containing the list of backups
    """
    
    try:
        response = await mcss_client.get(f"/servers/{server_id}/backups")
        
        if not response or (isinstance(response, list) and len(response) == 0):
            return "No backups found for this server."
        
        result = []
        for backup in response:
            backup_info = (
                f"Backup ID: {backup.get('backupId')}\n"
                f"Name: {backup.get('name')}\n"
                f"Destination: {backup.get('destination')}\n"
                f"Suspended: {backup.get('suspend')}\n"
                f"Delete Old Backups: {backup.get('deleteOldBackups')}\n"
                f"Last Status: {backup.get('lastStatus')}\n"
                f"Completed At: {backup.get('completedAt')}\n"
            )
            result.append(backup_info)
        
        return "\n---\n".join(result)
    except Exception as e:
        return f"Error fetching server backups: {str(e)}"


@mcp.tool()
async def get_backup_details(server_id: str, backup_id: str) -> str:
    """Get detailed information about a specific backup.
    
    Args:
        server_id: The ID of the server the backup belongs to
        backup_id: The ID of the backup to get details for
    
    Returns:
        A formatted string containing detailed information about the backup
    """
    
    try:
        response = await mcss_client.get(f"/servers/{server_id}/backups/{backup_id}")
        
        if not response:
            return f"No backup found with ID {backup_id}."
        
        # Format file blacklist
        file_blacklist = response.get('fileBlacklist', [])
        file_blacklist_str = "\n  - ".join(file_blacklist) if file_blacklist else "None"
        
        # Format folder blacklist
        folder_blacklist = response.get('folderBlacklist', [])
        folder_blacklist_str = "\n  - ".join(folder_blacklist) if folder_blacklist else "None"
        
        # Get compression type
        compression_types = {
            0: "None",
            1: "Fastest",
            2: "Fast",
            3: "Normal",
            4: "Maximum",
            5: "Ultra"
        }
        compression = response.get('compression', 0)
        compression_str = compression_types.get(compression, f"Unknown ({compression})")
        
        # Get last status
        status_types = {
            0: "None",
            1: "Running",
            2: "Completed",
            3: "Failed"
        }
        last_status = response.get('lastStatus', 0)
        status_str = status_types.get(last_status, f"Unknown ({last_status})")
        
        backup_details = (
            f"Backup ID: {response.get('backupId')}\n"
            f"Name: {response.get('name')}\n"
            f"Destination: {response.get('destination')}\n"
            f"Suspended: {response.get('suspend')}\n"
            f"Delete Old Backups: {response.get('deleteOldBackups')}\n"
            f"Compression: {compression_str}\n"
            f"Last Status: {status_str}\n"
            f"Completed At: {response.get('completedAt')}\n"
            f"File Blacklist:\n  - {file_blacklist_str}\n"
            f"Folder Blacklist:\n  - {folder_blacklist_str}\n"
        )
        
        return backup_details
    except Exception as e:
        return f"Error fetching backup details: {str(e)}"


@mcp.tool()
async def run_backup(server_id: str, backup_id: str) -> str:
    """Run a specific backup for a Minecraft server.
    
    Args:
        server_id: The ID of the server the backup belongs to
        backup_id: The ID of the backup to run
    
    Returns:
        A message indicating the result of the backup operation
    """
    
    try:
        # The correct endpoint is a POST to /servers/{server_id}/backups/{backup_id}
        response = await mcss_client.post(f"/servers/{server_id}/backups/{backup_id}", {})
        
        # If the response is empty or None, consider it a success
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully started backup {backup_id} for server {server_id}."
        
        # If we got a response, check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully started backup {backup_id} for server {server_id}."
        else:
            return f"Failed to run backup {backup_id} for server {server_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error running backup: {str(e)}"


@mcp.tool()
async def create_backup(server_id: str, name: str, destination: str, delete_old_backups: bool = False, 
                       compression: int = 0, file_blacklist: list = None, folder_blacklist: list = None) -> str:
    """Create a new backup for a Minecraft server.
    
    Args:
        server_id: The ID of the server to create a backup for
        name: The name of the backup
        destination: The destination folder for the backup
        delete_old_backups: Whether to delete old backups
        compression: The compression level (0=None, 1=Fastest, 2=Fast, 3=Normal, 4=Maximum, 5=Ultra)
        file_blacklist: List of files to exclude from the backup
        folder_blacklist: List of folders to exclude from the backup
    
    Returns:
        A message indicating the result of the backup creation
    """
    
    try:
        # Prepare the backup data
        backup_data = {
            "name": name,
            "destination": destination,
            "suspend": False,
            "deleteOldBackups": delete_old_backups,
            "compression": compression,
            "fileBlacklist": file_blacklist or [],
            "folderBlacklist": folder_blacklist or []
        }
        
        response = await mcss_client.post(f"/servers/{server_id}/backups", backup_data)
        
        if response and "backupId" in response:
            return f"Successfully created backup '{name}' with ID {response.get('backupId')}."
        else:
            return f"Failed to create backup. Response: {response}"
    except Exception as e:
        return f"Error creating backup: {str(e)}"


@mcp.tool()
async def delete_backup(server_id: str, backup_id: str) -> str:
    """Delete a backup for a Minecraft server.
    
    Args:
        server_id: The ID of the server the backup belongs to
        backup_id: The ID of the backup to delete
    
    Returns:
        A message indicating the result of the deletion
    """
    
    try:
        response = await mcss_client.delete(f"/servers/{server_id}/backups/{backup_id}")
        
        # If the response is empty or None, consider it a success
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully deleted backup {backup_id} from server {server_id}."
        
        # If we got a response, check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully deleted backup {backup_id} from server {server_id}."
        else:
            return f"Failed to delete backup {backup_id} from server {server_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error deleting backup: {str(e)}"


@mcp.tool()
async def set_api_credentials(host: str = None, port: int = None, api_key: str = None) -> str:
    """Set the API credentials for the MCSS API.
    
    Args:
        host: The hostname or IP address of the MCSS server
        port: The port of the MCSS web panel
        api_key: The API key for authentication
    
    Returns:
        A message indicating that the credentials have been set
    """
    global mcss_client
    mcss_client = MCSSClient(host, port, api_key)
    
    # Save the credentials to environment variables for this session
    os.environ["MCSS_HOST"] = host or os.environ.get("MCSS_HOST", DEFAULT_HOST)
    os.environ["MCSS_PORT"] = str(port or int(os.environ.get("MCSS_PORT", DEFAULT_PORT)))
    os.environ["MCSS_API_KEY"] = api_key or os.environ.get("MCSS_API_KEY", API_KEY)
    
    return f"API credentials set for {host or os.environ.get('MCSS_HOST', DEFAULT_HOST)}:{port or int(os.environ.get('MCSS_PORT', DEFAULT_PORT))}"


@mcp.tool()
async def get_scheduler_tasks(server_id: str) -> str:
    """Get a list of all scheduled tasks for a specific server.
    
    Args:
        server_id: The ID of the server to get tasks for
    
    Returns:
        A formatted string containing information about all scheduled tasks
    """
    try:
        response = await mcss_client.get(f"/servers/{server_id}/scheduler/tasks")
        tasks = response  # The response is already a list of tasks
        
        if not tasks:
            return f"No scheduled tasks found for server {server_id}."
        
        result = []
        for task in tasks:
            # Format timestamps for better readability
            last_run = task.get('lastRun', 'Never')
            next_run = task.get('nextRun', 'Not scheduled')
            
            # Try to format the timestamps if they exist
            if last_run and last_run != 'Never':
                try:
                    # Parse ISO format timestamp and format it more readably
                    last_run_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                    last_run = last_run_dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass  # Keep original format if parsing fails
            
            if next_run and next_run != 'Not scheduled':
                try:
                    next_run_dt = datetime.fromisoformat(next_run.replace('Z', '+00:00'))
                    next_run = next_run_dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass  # Keep original format if parsing fails
            
            # Extract job information
            jobs_info = ""
            jobs = task.get('jobs', [])
            if jobs:
                job_details = []
                for job in jobs:
                    if 'commands' in job:
                        job_details.append(f"Commands: {', '.join(job['commands'])}")
                    elif 'action' in job:
                        action_map = {0: "Stop", 1: "Start", 2: "Kill", 3: "Restart"}
                        action = action_map.get(job['action'], f"Unknown ({job['action']})")
                        job_details.append(f"Action: {action}")
                    elif 'backupIdentifier' in job:
                        job_details.append(f"Backup ID: {job['backupIdentifier']}")
                
                if job_details:
                    jobs_info = "\n" + "\n".join(job_details)
            
            # Extract timing information
            timing_info = ""
            timing = task.get('timing', {})
            if timing:
                if 'interval' in timing:
                    interval = timing['interval']
                    repeat = timing.get('repeat', False)
                    timing_info = f"\nInterval: {interval} seconds, Repeats: {'Yes' if repeat else 'No'}"
                elif 'timeSpan' in timing:
                    time_span = timing['timeSpan']
                    repeat = timing.get('repeat', False)
                    timing_info = f"\nTime Span: {time_span}, Repeats: {'Yes' if repeat else 'No'}"
            
            task_info = (
                f"Task ID: {task.get('id')}\n"
                f"Name: {task.get('name')}\n"
                f"Enabled: {'Yes' if task.get('enabled', False) else 'No'}"
                f"{timing_info}"
                f"\nLast Run: {last_run}\n"
                f"Next Run: {next_run}"
                f"{jobs_info}\n"
            )
            result.append(task_info)
        
        return "\n---\n".join(result)
    except Exception as e:
        return f"Error fetching scheduler tasks: {str(e)}"


@mcp.tool()
async def get_scheduler_task_details(server_id: str, task_id: str) -> str:
    """Get detailed information about a specific scheduled task.
    
    Args:
        server_id: The ID of the server the task belongs to
        task_id: The ID of the task to get details for
    
    Returns:
        A formatted string containing detailed information about the task
    """
    try:
        response = await mcss_client.get(f"/servers/{server_id}/scheduler/tasks/{task_id}")
        task = response  # The response is the task details
        
        if not task:
            return "No task details found."
        
        # Extract the task details
        details = [
            f"Task ID: {task.get('id', 'N/A')}",
            f"Name: {task.get('name', 'N/A')}",
            f"Enabled: {'Yes' if task.get('enabled', False) else 'No'}",
            f"Type: {task.get('type', 'N/A')}",
            f"Interval: {task.get('interval', 'N/A')}",
            f"Last Run: {task.get('lastRun', 'Never')}",
            f"Next Run: {task.get('nextRun', 'Not scheduled')}"
        ]
        
        # Add action details based on task type
        task_type = task.get("type")
        
        if task_type == "Command":
            details.append(f"Command: {task.get('command', 'N/A')}")
        elif task_type == "Backup":
            details.append(f"Backup ID: {task.get('backupId', 'N/A')}")
        elif task_type == "ServerAction":
            details.append(f"Action: {task.get('action', 'N/A')}")
        
        return "\n".join(details)
    except Exception as e:
        return f"Error getting task details: {str(e)}"


@mcp.tool()
async def create_scheduler_task(
    server_id: str,
    name: str,
    enabled: bool = True,
    player_requirement: int = 0,
    timing_type: str = "interval",
    repeat: bool = True,
    interval: int = 3600,
    job_type: str = "RunCommands",
    commands: list = None,
    backup_id: str = None,
    action: int = None
) -> str:
    """Create a new scheduled task for a server.
    
    Args:
        server_id: The ID of the server to create the task for
        name: The name of the task
        enabled: Whether the task is enabled
        player_requirement: Player requirement (0=none, 1=empty, 2=atLeastOne)
        timing_type: Type of timing (interval, fixedTime, timeless)
        repeat: Whether the task repeats
        interval: Interval in seconds (for interval timing)
        job_type: Type of job (ServerAction, RunCommands, StartBackup)
        commands: List of commands to run (for RunCommands job)
        backup_id: ID of the backup to run (for StartBackup job)
        action: Server action to perform (for ServerAction job, 0=Stop, 1=Start, 2=Kill, 3=Restart)
    
    Returns:
        A message indicating the result of the task creation
    """
    try:
        # Create timing object based on timing_type
        timing = {}
        if timing_type == "interval":
            timing = {
                "repeat": repeat,
                "interval": interval
            }
        elif timing_type == "fixedTime":
            timing = {
                "repeat": repeat,
                "timeSpan": interval
            }
        elif timing_type == "timeless":
            timing = {}  # Timeless has no parameters
        
        # Create jobs array based on job_type
        jobs = []
        
        # Convert action string to number if provided as string
        if isinstance(action, str):
            action_map = {"stop": 0, "start": 1, "kill": 2, "restart": 3}
            if action.lower() in action_map:
                action = action_map[action.lower()]
        
        if job_type == "RunCommands":
            jobs.append({
                "order": 0,
                "commands": commands or []
            })
        elif job_type == "StartBackup":
            jobs.append({
                "order": 0,
                "backupIdentifier": backup_id
            })
        elif job_type == "ServerAction":
            jobs.append({
                "order": 0,
                "action": action
            })
        
        # Create the task data
        task_data = {
            "name": name,
            "enabled": enabled,
            "playerRequirement": player_requirement,
            "timing": timing,
            "jobs": jobs
        }
        
        # Send the request to create the task
        response = await mcss_client.post(f"/servers/{server_id}/scheduler/tasks", task_data)
        
        # Check if we got a valid response
        if response is None:
            return "Failed to create task. No response from server."
        
        # Extract task ID from response if available
        task_id = None
        if isinstance(response, dict):
            task_id = response.get("id")
        
        if task_id:
            return f"Successfully created task: {name} (Task ID: {task_id})"
        else:
            return f"Task created successfully, but no task ID was returned in the response."
    except Exception as e:
        return f"Error creating task: {str(e)}"


@mcp.tool()
async def update_scheduler_task(
    server_id: str,
    task_id: str,
    name: str = None,
    enabled: bool = None,
    player_requirement: int = None,
    timing_type: str = None,
    repeat: bool = None,
    interval: int = None,
    job_type: str = None,
    commands: list = None,
    backup_id: str = None,
    action: str = None
) -> str:
    """Update an existing scheduled task.
    
    Args:
        server_id: The ID of the server the task belongs to
        task_id: The ID of the task to update
        name: The new name for the task
        enabled: Whether the task is enabled
        player_requirement: Player requirement (0=none, 1=empty, 2=atLeastOne)
        timing_type: Type of timing (interval, fixedTime, timeless)
        repeat: Whether the task repeats
        interval: Interval in seconds (for interval timing)
        job_type: Type of job (ServerAction, RunCommands, StartBackup)
        commands: List of commands to run (for RunCommands job)
        backup_id: ID of the backup to run (for StartBackup job)
        action: Server action to perform (for ServerAction job, 0=Stop, 1=Start, 2=Kill, 3=Restart)
    
    Returns:
        A message indicating the result of the task update
    """
    try:
        # First, get the current task details
        current_task = await mcss_client.get(f"/servers/{server_id}/scheduler/tasks/{task_id}")
        
        if not current_task:
            return f"Task with ID {task_id} not found for server {server_id}."
        
        # Create the update data with only the fields that are provided
        update_data = {}
        
        if name is not None:
            update_data["name"] = name
        else:
            update_data["name"] = current_task.get("name", "")
        
        if enabled is not None:
            update_data["enabled"] = enabled
        else:
            update_data["enabled"] = current_task.get("enabled", True)
        
        if player_requirement is not None:
            update_data["playerRequirement"] = player_requirement
        else:
            update_data["playerRequirement"] = current_task.get("playerRequirement", 0)
        
        # Handle timing updates
        current_timing = current_task.get("timing", {})
        timing = {}
        
        # Determine the timing type from the current task if not provided
        if timing_type is None:
            # Try to infer timing type from current task
            if "interval" in current_timing:
                timing_type = "interval"
            elif "timeSpan" in current_timing:
                timing_type = "fixedTime"
            else:
                timing_type = "timeless"
        
        if timing_type == "interval":
            timing["repeat"] = repeat if repeat is not None else current_timing.get("repeat", True)
            timing["interval"] = interval if interval is not None else current_timing.get("interval", 3600)
        elif timing_type == "fixedTime":
            timing["repeat"] = repeat if repeat is not None else current_timing.get("repeat", True)
            timing["timeSpan"] = interval if interval is not None else current_timing.get("timeSpan", 3600)
        elif timing_type == "timeless":
            timing = {}  # Timeless has no parameters
        
        update_data["timing"] = timing
        
        # Handle jobs updates
        current_jobs = current_task.get("jobs", [])
        jobs = []
        
        # If no job type is provided, keep the existing jobs
        if job_type is None:
            jobs = current_jobs
        else:
            # Convert action string to number if provided as string
            action_map = {"stop": 0, "start": 1, "kill": 2, "restart": 3}
            if action is not None and isinstance(action, str) and action.lower() in action_map:
                action = action_map[action.lower()]
            
            if job_type == "RunCommands":
                jobs.append({
                    "order": 0,
                    "commands": commands or []
                })
            elif job_type == "StartBackup":
                jobs.append({
                    "order": 0,
                    "backupIdentifier": backup_id
                })
            elif job_type == "ServerAction":
                jobs.append({
                    "order": 0,
                    "action": action
                })
        
        update_data["jobs"] = jobs
        
        # Only update if there are changes
        if not update_data:
            return "No changes to update."
        
        response = await mcss_client.post(f"/servers/{server_id}/scheduler/tasks/{task_id}", update_data)
        
        # Check if we got a valid response
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully updated task with ID: {task_id}"
        
        # Check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully updated task with ID: {task_id}"
        else:
            return f"Failed to update task with ID: {task_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error updating task: {str(e)}"


@mcp.tool()
async def delete_scheduler_task(server_id: str, task_id: str) -> str:
    """Delete a scheduled task.
    
    Args:
        server_id: The ID of the server the task belongs to
        task_id: The ID of the task to delete
    
    Returns:
        A message indicating the result of the task deletion
    """
    try:
        response = await mcss_client.delete(f"/servers/{server_id}/scheduler/tasks/{task_id}")
        
        # If the response is empty or None, consider it a success
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully deleted task with ID: {task_id}"
        
        # Check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully deleted task with ID: {task_id}"
        else:
            return f"Failed to delete task with ID: {task_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error deleting task: {str(e)}"


@mcp.tool()
async def run_scheduler_task(server_id: str, task_id: str) -> str:
    """Run a scheduled task immediately.
    
    Args:
        server_id: The ID of the server the task belongs to
        task_id: The ID of the task to run
    
    Returns:
        A message indicating the result of running the task
    """
    try:
        response = await mcss_client.post(f"/servers/{server_id}/scheduler/tasks/{task_id}/run", {})
        
        # If the response is empty or None, consider it a success
        if response is None or (isinstance(response, dict) and not response):
            return f"Successfully ran task with ID: {task_id}"
        
        # Check for success or error
        success = response.get("success", True)  # Default to True if no success field
        
        if success:
            return f"Successfully ran task with ID: {task_id}"
        else:
            return f"Failed to run task with ID: {task_id}. Error: {response.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error running task: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
