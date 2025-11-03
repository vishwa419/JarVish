#!/bin/bash
set -e

echo "=========================================="
echo "RAG MCP Server Startup"
echo "=========================================="

# Display configuration
echo "Configuration:"
echo "  QDRANT_HOST: ${QDRANT_HOST}"
echo "  QDRANT_PORT: ${QDRANT_PORT}"
echo "  QDRANT_COLLECTION: ${QDRANT_COLLECTION}"
echo "  EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
echo "  CHUNK_SIZE: ${CHUNK_SIZE}"
echo "  CHUNK_OVERLAP: ${CHUNK_OVERLAP}"
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY is not set!"
    exit 1
fi
echo "✅ OpenAI API key is configured"

# Wait for Qdrant to be ready
echo ""
echo "🔍 Waiting for Qdrant at ${QDRANT_HOST}:${QDRANT_PORT}..."

MAX_TRIES=30
COUNT=0
until curl -sf http://${QDRANT_HOST}:${QDRANT_PORT}/healthz > /dev/null 2>&1; do
  COUNT=$((COUNT+1))
  if [ $COUNT -ge $MAX_TRIES ]; then
    echo "❌ Qdrant failed to start after 60 seconds"
    echo "   Please check: docker-compose logs qdrant"
    exit 1
  fi
  echo "⏳ Waiting for Qdrant... (attempt $COUNT/$MAX_TRIES)"
  sleep 2
done

echo "✅ Qdrant is ready!"
echo ""

# Start FastMCP server
echo "=========================================="
echo "🚀 Starting RAG MCP Server on port 8001"
echo "=========================================="
echo ""

# Use fastmcp CLI to run the server
exec fastmcp run server.py --transport http --host 0.0.0.0 --port 8001
