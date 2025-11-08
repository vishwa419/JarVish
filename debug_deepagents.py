# debug_deepagents.py
"""
Debug script to understand deepagents structure
"""
import inspect
from deepagents import create_deep_agent

# Inspect the create_deep_agent function signature
print("=== create_deep_agent signature ===")
sig = inspect.signature(create_deep_agent)
print(sig)
print()

# Check the docstring
print("=== create_deep_agent docstring ===")
print(inspect.getdoc(create_deep_agent))
print()

# Try to see what parameters it expects
print("=== Parameters ===")
for param_name, param in sig.parameters.items():
    print(f"{param_name}: {param.annotation} = {param.default}")
print()

# Try creating a minimal agent to see the structure
from langchain_openai import ChatOpenAI

print("=== Creating minimal test agent ===")
try:
    minimal_agent = create_deep_agent(
        name="test",
        model=ChatOpenAI(model="gpt-4o-mini"),
        system_prompt="You are a test agent.",
        subagents=[
            {
                "name": "worker1",
                "agent_type": "general-purpose",
                "model": ChatOpenAI(model="gpt-4o-mini"),
                "description": "A worker agent",
                "system_prompt": "You are worker 1"
            },
            {
                "name": "synthesizer",
                "agent_type": "synthesizer",
                "model": ChatOpenAI(model="gpt-4o-mini"),
                "description": "Synthesizer",
                "system_prompt": "You synthesize results"
            }
        ]
    )
    
    print("✅ Minimal agent created successfully!")
    print(f"Agent type: {type(minimal_agent)}")
    print(f"Agent attributes: {[a for a in dir(minimal_agent) if not a.startswith('_')]}")
    
    # Check if subagents were registered correctly
    if hasattr(minimal_agent, 'config'):
        print(f"\nAgent config: {minimal_agent.config}")
    
    if hasattr(minimal_agent, 'get_graph'):
        print("\n=== Agent Graph ===")
        print(minimal_agent.get_graph().draw_ascii())
        
except Exception as e:
    print(f"❌ Failed to create minimal agent: {e}")
    import traceback
    traceback.print_exc()
