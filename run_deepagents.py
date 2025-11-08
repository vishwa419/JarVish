# run_deepagents.py
from backend.deep_chain import create_jarvis_deep_agent
from backend.models import MCPServerConfig
from langchain_core.messages import HumanMessage

servers = [
    MCPServerConfig(name="gmail", url="http://localhost:8001", description="Email MCP"),
    MCPServerConfig(name="calendar", url="http://localhost:8002", description="Calendar MCP"),
]

jarvis = create_jarvis_deep_agent(servers)

# ✅ Use messages format (this is what deepagents expects)
response = jarvis.invoke({
    "messages": [
        HumanMessage(content="Find Sathvika's latest email from qdrant. schedule a meeting tomorrow. at 8:30pm, for 30 mins")
    ]
})

print("=== Full Response ===")
print(response)

# Extract the final answer
if "messages" in response:
    final_message = response["messages"][-1]
    print("\n=== Final Answer ===")
    print(final_message.content)
