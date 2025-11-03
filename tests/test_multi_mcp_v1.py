"""
Integration tests for Multi-MCP Jarvis Agent
Tests the full stack: FastAPI -> JarvisChain -> Multi-MCP -> RAG Server
"""
import asyncio
import sys
import httpx
from pathlib import Path

# Test configuration
FASTAPI_URL = "http://localhost:8000"
TEST_DOCUMENT_CONTENT = """
Quantum computing represents a paradigm shift in computational capabilities.
Unlike classical computers that use bits, quantum computers use qubits.
Qubits can exist in superposition, allowing quantum computers to process multiple states simultaneously.
Major tech companies like IBM, Google, and Microsoft are investing heavily in quantum research.
Quantum supremacy was demonstrated by Google in 2019.
"""


async def setup_test_document():
    """Create a test document and upload it to the RAG server."""
    uploads_dir = Path("./data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = uploads_dir/"quantum_computing.txt"
    with open(test_file, 'w') as f:
        f.write(TEST_DOCUMENT_CONTENT)
    
    print(f"✅ Created test document: {test_file}")
    return "quantum_computing.txt"


async def test_health_check():
    """Test: Check if FastAPI server is healthy."""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FASTAPI_URL}/health")
            data = response.json()
            
            print(f"Status: {data['status']}")
            print(f"Jarvis Ready: {data['jarvis_ready']}")
            print(f"MCP Servers: {data['mcp_servers']}")
            print(f"Total Tools: {data['total_tools']}")
            
            if data['jarvis_ready'] and data['total_tools'] > 0:
                print("✅ Health check passed")
                return True
            else:
                print("❌ Health check failed")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_list_tools():
    """Test: List all available tools."""
    print("\n" + "="*70)
    print("TEST 2: List Tools")
    print("="*70)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FASTAPI_URL}/tools")
            data = response.json()
            
            tools = data['tools']
            server_info = data['server_info']
            
            print(f"Found {len(tools)} tools from {len(server_info['servers'])} servers:\n")
            
            for server in server_info['servers']:
                print(f"📡 Server: {server['name']}")
                print(f"   URL: {server['url']}")
                print(f"   Description: {server['description']}")
            
            print(f"\n📋 Available tools:")
            for tool in tools:
                print(f"   • {tool}")
            
            # Check for expected RAG tools
            expected_tools = ['qdrant_store', 'qdrant_search', 'list_documents']
            has_rag_tools = all(tool in tools for tool in expected_tools)
            
            if has_rag_tools:
                print("\n✅ All expected RAG tools found")
                return True
            else:
                print("\n❌ Missing expected RAG tools")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_store_document():
    """Test: Store document via Jarvis agent."""
    print("\n" + "="*70)
    print("TEST 3: Store Document via Agent")
    print("="*70)
    
    try:
        filename = await setup_test_document()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Ask Jarvis to store the document
            response = await client.post(
                f"{FASTAPI_URL}/chat",
                data={"message": f"Please index the document ./data/uploads/{filename}"}
            )
            
            html_response = response.text
            print(f"Agent Response:\n{html_response}\n")
            
            # Check if response indicates success
            if "✅" in html_response or "success" in html_response.lower():
                print("✅ Document stored successfully")
                return True
            else:
                print("❌ Document storage may have failed")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_search_document():
    """Test: Search documents via Jarvis agent."""
    print("\n" + "="*70)
    print("TEST 4: Search Documents via Agent")
    print("="*70)
    
    queries = [
        "What is quantum computing?",
        "Tell me about qubits",
        "Which companies are investing in quantum research?"
    ]
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for query in queries:
                print(f"\n🔍 Query: {query}")
                
                response = await client.post(
                    f"{FASTAPI_URL}/chat",
                    data={"message": query}
                )
                
                html_response = response.text
                
                # Extract assistant response (simplified parsing)
                if "assistant-message" in html_response:
                    # Simple extraction - in production you'd use proper HTML parsing
                    print(f"✅ Got response from agent")
                else:
                    print(f"❌ No response from agent")
                
                print("-" * 70)
        
        print("\n✅ Search tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_list_documents():
    """Test: List documents via Jarvis agent."""
    print("\n" + "="*70)
    print("TEST 5: List Documents via Agent")
    print("="*70)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/chat",
                data={"message": "What documents do you have indexed?"}
            )
            
            html_response = response.text
            print(f"Agent Response:\n{html_response}\n")
            
            if "quantum_computing" in html_response.lower():
                print("✅ Document listing successful")
                return True
            else:
                print("❌ Document not found in listing")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_multi_tool_query():
    """Test: Complex query requiring multiple tool calls."""
    print("\n" + "="*70)
    print("TEST 6: Multi-Tool Complex Query")
    print("="*70)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Ask a question that requires both listing and searching
            response = await client.post(
                f"{FASTAPI_URL}/chat",
                data={
                    "message": "Show me all the documents you have, then tell me about quantum supremacy"
                }
            )
            
            html_response = response.text
            print(f"Agent Response:\n{html_response}\n")
            
            # Check if agent used multiple tools
            if "quantum" in html_response.lower() and "supremacy" in html_response.lower():
                print("✅ Multi-tool query successful")
                return True
            else:
                print("❌ Multi-tool query incomplete")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("🧪 MULTI-MCP INTEGRATION TESTS - FULL STACK")
    print("="*70)
    print("Testing: FastAPI -> JarvisChain -> Multi-MCP -> RAG Server")
    print("="*70)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("List Tools", test_list_tools),
        ("Store Document", test_store_document),
        ("Search Documents", test_search_document),
        ("List Documents", test_list_documents),
        ("Multi-Tool Query", test_multi_tool_query),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    print("\n⚠️  Prerequisites:")
    print("   1. docker-compose up -d (Qdrant + RAG MCP server running)")
    print("   2. uvicorn backend.main_updated:app --reload (FastAPI server running)")
    print("   3. OPENAI_API_KEY set in .env")
    print("\nStarting tests in 3 seconds...\n")
    
    import time
    time.sleep(3)
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
