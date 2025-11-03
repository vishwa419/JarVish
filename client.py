# client.py
from google_workspace_mcp import MCPClient

# Point to your MCP server
MCP_URL = "http://localhost:8001/mcp"

# Create client
client = MCPClient(MCP_URL)

# Call Gmail API (list labels)
labels = client.gmail.list_labels()
print("Your Gmail Labels:")
for label in labels:
    print("-", label["name"])

