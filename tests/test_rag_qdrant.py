"""
Tests for RAG MCP Server with Qdrant
Run these tests after docker-compose is up.
"""
import asyncio
import sys
from pathlib import Path
from fastmcp import Client

# Test configuration
RAG_MCP_URL = "http://localhost:8001/mcp"  # HTTP transport
TEST_FILE = "test_document1.txt"
TEST_CONTENT = """
Artificial Intelligence (AI) is transforming the world.
Machine learning is a subset of AI focused on learning from data.
Natural Language Processing enables computers to understand human language.
Computer vision allows machines to interpret visual information.
Deep learning uses neural networks with multiple layers.
"""


async def setup_test_document():
    """Create a test document in uploads directory."""
    uploads_dir = Path("./data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    test_file_path = uploads_dir / TEST_FILE
    with open(test_file_path, 'w') as f:
        f.write(TEST_CONTENT)
    
    print(f"✅ Created test document: {test_file_path}")


async def test_list_tools():
    """Test: List available tools from RAG MCP server."""
    print("\n" + "="*60)
    print("TEST 1: List Tools")
    print("="*60)
    
    try:
        async with Client(RAG_MCP_URL) as client:
            result = await client.list_tools()
            tools = result.tools if hasattr(result, 'tools') else result
            
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                name = tool.name if hasattr(tool, 'name') else tool['name']
                desc = tool.description if hasattr(tool, 'description') else tool.get('description', '')
                print(f"   • {name}: {desc}")
            
            return True
    except Exception as e:
        print(f"❌ Error listing tools: {e}")
        return False


async def test_store_document():
    """Test: Store test document in Qdrant."""
    print("\n" + "="*60)
    print("TEST 2: Store Document")
    print("="*60)
    
    try:
        async with Client(RAG_MCP_URL) as client:
            result = await client.call_tool(
                "qdrant_store",
                {"filename": TEST_FILE, "force_reload": True}
            )
            
            # Extract response text
            if hasattr(result, 'content'):
                response = result.content[0].text if isinstance(result.content, list) else str(result.content)
            else:
                response = str(result)
            
            print(f"Response: {response}")
            
            if "✅" in response:
                print("✅ Document stored successfully")
                return True
            else:
                print("❌ Document storage failed")
                return False
                
    except Exception as e:
        print(f"❌ Error storing document: {e}")
        return False


async def test_search_documents():
    """Test: Search for content in stored documents."""
    print("\n" + "="*60)
    print("TEST 3: Search Documents")
    print("="*60)
    
    queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How does computer vision work?"
    ]
    
    try:
        async with Client(RAG_MCP_URL) as client:
            for query in queries:
                print(f"\n🔍 Query: {query}")
                
                result = await client.call_tool(
                    "qdrant_search",
                    {"query": query, "top_k": 3}
                )
                
                # Extract response text
                if hasattr(result, 'content'):
                    response = result.content[0].text if isinstance(result.content, list) else str(result.content)
                else:
                    response = str(result)
                
                print(f"Results:\n{response}")
                print("-" * 60)
        
        print("✅ Search tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return False


async def test_list_documents():
    """Test: List all indexed documents."""
    print("\n" + "="*60)
    print("TEST 4: List Documents")
    print("="*60)
    
    try:
        async with Client(RAG_MCP_URL) as client:
            result = await client.call_tool("list_documents", {})
            
            # Extract response text
            if hasattr(result, 'content'):
                response = result.content[0].text if isinstance(result.content, list) else str(result.content)
            else:
                response = str(result)
            
            print(f"Indexed documents:\n{response}")
            print("✅ List documents test completed")
            return True
            
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        return False


async def test_reload_all():
    """Test: Reload all documents."""
    print("\n" + "="*60)
    print("TEST 5: Reload All Documents")
    print("="*60)
    
    try:
        async with Client(RAG_MCP_URL) as client:
            result = await client.call_tool("reload_all_documents", {})
            
            # Extract response text
            if hasattr(result, 'content'):
                response = result.content[0].text if isinstance(result.content, list) else str(result.content)
            else:
                response = str(result)
            
            print(f"Reload result:\n{response}")
            print("✅ Reload test completed")
            return True
            
    except Exception as e:
        print(f"❌ Error reloading: {e}")
        return False


async def run_all_tests():
    """Run all RAG tests."""
    print("\n" + "="*70)
    print("🧪 RAG MCP SERVER - QDRANT INTEGRATION TESTS")
    print("="*70)
    
    # Setup
    await setup_test_document()
    
    # Run tests
    tests = [
        ("List Tools", test_list_tools),
        ("Store Document", test_store_document),
        ("Search Documents", test_search_documents),
        ("List Documents", test_list_documents),
        ("Reload All", test_reload_all),
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
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
