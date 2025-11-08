# run_langgraph.py
from backend.langgraph_agents.orchestrator import create_orchestrator
from backend.models import MCPServerConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

print("=" * 80)
print("🤖 Initializing Jarvis LangGraph Agent System (No DeepAgents)")
print("=" * 80)

# ---------------------------------------------------------------------
# CONFIGURE MCP SERVERS
# ---------------------------------------------------------------------
servers = [
    MCPServerConfig(name="qdrant_store", url="http://localhost:8001", description="Qdrant MCP Server"),
    MCPServerConfig(name="google_workspace", url="http://localhost:8002", description="Google Workspace MCP Server"),
]

print("\n📡 MCP Servers configured:")
for server in servers:
    print(f"  - {server.name}: {server.url}")

# ---------------------------------------------------------------------
# BUILD ORCHESTRATOR GRAPH
# ---------------------------------------------------------------------
print("\n🔧 Building LangGraph orchestrator...")
jarvis = create_orchestrator(servers)

# ---------------------------------------------------------------------
# RUNTIME CONFIG
# ---------------------------------------------------------------------
RECURSION_LIMIT = 10
memory = MemorySaver()  # store reasoning checkpoints
config = {
    "recursion_limit": RECURSION_LIMIT,
    "checkpoint": memory,
}

# ---------------------------------------------------------------------
# DEBUG UTILITIES
# ---------------------------------------------------------------------
def log_step(agent_name: str, message: str):
    """Prints timestamped structured trace."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{agent_name.upper()}] {message}")

# Monkeypatch or hook your orchestrator/subagent logic to call log_step
# Example (if you have hooks inside create_orchestrator):
#   subagent_node() or plan_node() can call: log_step("qdrant_store", "Executing vector upsert...")

# ---------------------------------------------------------------------
# EXECUTE QUERY
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("📝 Processing Query")
print("=" * 80)

query = "my email: kafkafranz495@gmail.com. Summarize about AI using qdrant_search. use qdrant_search to find sathvika's email. Use that email and send her an email include the AI summary use google_workspace. schedule an event with google_workspace for nov 10, 2025, 11:30 PST, I am already authorized"
print(f"\nQuery: {query}")
print(f"⚙️ Recursion limit set to {RECURSION_LIMIT}\n")

try:
    result = jarvis.invoke({"messages": [HumanMessage(content=query)]}, config=config)
except Exception as e:
    print("\n❌ ERROR:")
    print(e)
    exit(1)

# ---------------------------------------------------------------------
# DISPLAY FINAL RESULT
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("✅ FINAL RESULT")
print("=" * 80)
print(result.get("final_answer", "⚠️ No final answer returned."))

# ---------------------------------------------------------------------
# OPTIONAL: PRINT MESSAGES
# ---------------------------------------------------------------------
if "messages" in result:
    print("\n" + "=" * 80)
    print("📋 Message History")
    print("=" * 80)
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"\n{i+1}. {msg_type}:")
        print(f"   {content}")

