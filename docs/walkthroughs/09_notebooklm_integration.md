# Walkthrough: NotebookLM Antigravity Integration

I have successfully connected Antigravity to your NotebookLM account. Your AI agent can now interact with your notebooks directly.

## Changes Made

### 1. Installed MCP Server
I used `uv` to install the `notebooklm-mcp-cli` server, which provides the bridge between Antigravity and the NotebookLM private API.

### 2. Configured Antigravity
Updated `~/.gemini/antigravity/mcp_config.json` with the path to the server binary:
```json
{
  "mcpServers": {
    "notebooklm": {
      "command": "/Users/kpakpo/.local/bin/notebooklm-mcp",
      "args": []
    }
  }
}
```

### 3. Verified Authentication
The `nlm login` process successfully captured the session cookies for `kpakpo@gmail.com`.

## Verification Results

I verified the connection by listing your notebooks via the CLI:
```bash
nlm notebook list
```
The command returned a table of your existing notebooks, confirming that the authentication is active and the tool is functional.

> [!NOTE]
> **Important:** If you don't see the new `notebooklm` tools available in Antigravity immediately, you may need to **restart the Antigravity application**. This forces the environment to reload the `mcp_config.json` and initialize the new server.

## How to Use
You can now ask Antigravity to perform tasks like:
- "List my NotebookLM notebooks."
- "Create a new notebook in NotebookLM about trading strategies."
- "Ask my 'Quant Research' notebook for a summary of the latest PDF source."

## Troubleshooting
If you experience "Profile not found" or "Unauthorized" errors in the future, simply run:
```bash
nlm login
```
to refresh your session cookies.
