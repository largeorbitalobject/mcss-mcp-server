# MCSS MCP Server

![MCSS MCP Server](https://img.shields.io/badge/MCSS-MCP%20Server-brightgreen)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-yellow)

A Model Context Protocol (MCP) server for controlling Minecraft servers via the MCSS (Minecraft Server Software) API. This tool enables seamless interaction with your Minecraft servers using MCP clients like Claude Desktop.

## 🌟 Features

- **Server Management**
  - List all Minecraft servers managed by MCSS
  - Get detailed information about specific servers
  - Execute power actions (start, stop, restart, kill)
  - Execute Minecraft commands
  - View server console output in real-time
  - Update server settings (name, description, crash handling, etc.)

- **Backup Management**
  - List available backups
  - Create new backup configurations
  - Run backups on demand

- **Scheduler Management**
  - List scheduled tasks
  - Create new scheduled tasks (commands, backups, server actions)
  - Update existing tasks
  - Delete tasks
  - Run tasks on demand

## 📋 Prerequisites

- Python 3.10 or higher
- MCSS (Minecraft Server Software) with the Web API enabled
- An API key for the MCSS Web API
- Claude Desktop (for MCP client functionality)

## 🚀 Quick Start

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/largeorbitalobject/mcss-mcp-server.git
   cd mcss-mcp-server
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install mcp[cli] httpx python-dotenv
   ```

### Configuration

1. Create a `.env` file in the project root directory:
   ```
   MCSS_HOST=your_mcss_host_ip
   MCSS_PORT=25560
   MCSS_API_KEY=your_mcss_api_key
   ```

2. Configure Claude Desktop to use this MCP server:
   - Open your Claude Desktop configuration file:
     - Windows: `%AppData%\Claude\claude_desktop_config.json`
     - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Add the following configuration:
     ```json
     {
       "mcpServers": {
         "mcss-control": {
           "command": "C:\\path\\to\\mcss-mcp-server\\run_mcss_mcp.bat",
           "args": []
         }
       }
     }
     ```
   - Replace the path with the actual path to the batch file on your system

## 🔧 Using the MCP Tools in Claude Desktop

Once Claude Desktop is launched, the server will start automatically and you can use the following tools:

### Server Management

```
get_servers()
```
Returns a list of all Minecraft servers managed by MCSS.

```
get_server_details(server_id)
```
Returns detailed information about a specific server.

```
edit_server(server_id, name=None, description=None, is_set_to_auto_start=None, force_save_on_stop=None, java_allocated_memory=None, keep_online=None)
```
Updates a specific Minecraft server's settings. The `keep_online` parameter controls crash handling (0=none, 1=elevated, 2=aggressive).

```
execute_server_action(server_id, action)
```
Executes a power action on a server. Valid actions: `start`, `stop`, `restart`, `kill`.

```
execute_server_command(server_id, command)
```
Executes a Minecraft command on a server.

```
get_server_console(server_id, lines=50)
```
Returns the console output of a server.

### Backup Management

```
get_backups(server_id)
```
Returns a list of backups for a server.

```
create_backup(server_id, name, description="", compression=None)
```
Creates a new backup configuration.

```
run_backup(server_id, backup_id)
```
Runs a backup for a server.

### Scheduler Management

```
get_scheduler_tasks(server_id)
```
Returns a list of scheduled tasks for a server.

```
create_scheduler_task(server_id, name, enabled=True, ...)
```
Creates a new scheduled task.

```
update_scheduler_task(server_id, task_id, name=None, ...)
```
Updates an existing scheduled task.

```
delete_scheduler_task(server_id, task_id)
```
Deletes a scheduled task.

```
run_scheduler_task(server_id, task_id)
```
Runs a scheduled task immediately.

## 📁 Project Structure

```
mcss-mcp-server/
├── mcss_mcp/                # Main package
│   ├── __init__.py          # Package initialization
│   └── server.py            # MCP server implementation
├── main.py                  # Entry point script
├── run_mcss_mcp.bat         # Batch script to run the server
├── .env                     # Environment variables (not in repo)
└── README.md                # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MCSS](https://mcserversoft.com/) for providing the Minecraft server management software
- [MCP](https://modelcontextprotocol.io/introduction) for the Model Context Protocol specification
- [Claude Desktop](https://claude.ai/download) for the MCP client implementation
