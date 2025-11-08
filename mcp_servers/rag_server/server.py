"""
RAG MCP Server - Qdrant-backed semantic search
Exposes tools for document ingestion and search via FastMCP.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid

from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import hashlib
import json

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Jarvis RAG Server")

# Configuration from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jarvis_documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
UPLOADS_DIR = Path("/app/uploads")

# Initialize clients
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def initialize_collection():
    """Create Qdrant collection if it doesn't exist."""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating collection: {QDRANT_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small dimensions
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Collection {QDRANT_COLLECTION} created")
        else:
            logger.info(f"✅ Collection {QDRANT_COLLECTION} already exists")
    except Exception as e:
        logger.error(f"❌ Error initializing collection: {e}")
        raise


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks based on token count approximation.
    
    Args:
        text: Text to chunk
        chunk_size: Approximate tokens per chunk
        overlap: Approximate tokens to overlap
        
    Returns:
        List of text chunks
    """
    # Rough approximation: 1 token ≈ 4 characters
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + char_chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_length:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > char_chunk_size * 0.5:  # At least 50% of chunk
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - char_overlap
    
    return [c for c in chunks if c]  # Filter empty chunks


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI API."""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def read_file_content(filepath: Path) -> str:
    """
    Read content from file based on extension.
    
    Args:
        filepath: Path to file
        
    Returns:
        Text content
    """
    ext = filepath.suffix.lower()
    
    if ext == '.txt' or ext == '.md':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == '.pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("pypdf not installed. Install with: pip install pypdf")
    
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def generate_document_id(filename: str) -> str:
    """Generate consistent document ID from filename."""
    return hashlib.md5(filename.encode()).hexdigest()


def generate_point_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate a valid UUID for Qdrant point ID.
    Uses MD5 hash of doc_id + chunk_index to create consistent UUID.
    """
    # Create a unique string for this chunk
    unique_str = f"{doc_id}_{chunk_index}"
    # Generate UUID from hash (UUID v5 using DNS namespace)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))


@mcp.tool()
async def qdrant_store(filename: str, force_reload: bool = False) -> str:
    """
    Ingest a document from uploads folder into Qdrant.
    
    Args:
        filename: Name of file in uploads directory
        force_reload: If True, re-index even if already exists
        
    Returns:
        Status message with number of chunks stored
    """
    try:
        query = json.loads(filename)
        filename = query['filename']
    except Exception as e:
        logger.info(f"Filename: {filename}, error: {e}")
    logger.info(f"Called Qdrant store with: {filename}")
    try:
        filepath = UPLOADS_DIR / filename
        
        if not filepath.exists():
            return f"❌ Error: File '{filename}' not found in uploads directory"
        
        logger.info(f"📄 Processing document: {filename}")
        
        # Generate document ID
        doc_id = generate_document_id(filename)
        
        # Check if document already exists
        if not force_reload:
            try:
                results = qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter={
                        "must": [
                            {"key": "document_id", "match": {"value": doc_id}}
                        ]
                    },
                    limit=1
                )
                if results[0]:
                    return f"ℹ️ Document '{filename}' already indexed. Use force_reload=true to re-index."
            except:
                pass  # Collection might be empty
        
        # Read file content
        content = read_file_content(filepath)
        logger.info(f"📖 Read {len(content)} characters from {filename}")
        
        # Chunk content
        chunks = chunk_text(content)
        logger.info(f"✂️ Split into {len(chunks)} chunks")
        
        # Generate embeddings and store
        points = []
        for idx, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            
            # Generate valid UUID for point ID
            point_id = generate_point_id(doc_id, idx)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk,
                    "token_count": len(chunk) // 4,  # Rough estimate
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "total_chunks": len(chunks)
                }
            )
            points.append(point)
        
        # Upsert to Qdrant
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        
        logger.info(f"✅ Stored {len(points)} chunks for {filename}")
        return f"✅ Successfully indexed '{filename}' ({len(chunks)} chunks)"
        
    except Exception as e:
        error_msg = f"❌ Error storing document: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def qdrant_search(query: str, top_k: int = 5) -> str:
    """
    Semantic search across all documents in Qdrant.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        Formatted search results with relevance scores
    """
    try:
        logger.info(f"🔍 Searching for: {query}")
        
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Search Qdrant
        results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k
        )
        
        if not results:
            return "No results found. Make sure documents are indexed using qdrant_store."
        
        # Format results
        output = f"Found {len(results)} relevant passages:\n\n"
        
        for i, hit in enumerate(results, 1):
            payload = hit.payload
            score = hit.score
            
            output += f"Result {i} (Score: {score:.3f}):\n"
            output += f"Source: {payload['filename']} (Chunk {payload['chunk_index'] + 1}/{payload['total_chunks']})\n"
            output += f"Text: {payload['text']}...\n\n"
        
        logger.info(f"✅ Returned {len(results)} results")
        return output
        
    except Exception as e:
        error_msg = f"❌ Error searching: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def reload_all_documents() -> str:
    """
    Re-index all documents in uploads folder.
    Useful after bulk uploads or Qdrant reset.
    
    Returns:
        Summary of indexed documents
    """
    try:
        logger.info("🔄 Reloading all documents...")
        
        # Get all files in uploads
        files = [f for f in UPLOADS_DIR.iterdir() if f.is_file()]
        
        if not files:
            return "No files found in uploads directory."
        
        results = []
        success_count = 0
        
        for filepath in files:
            result = await qdrant_store(filepath.name, force_reload=True)
            results.append(f"• {filepath.name}: {result}")
            if "✅" in result:
                success_count += 1
        
        summary = f"✅ Reloaded {success_count}/{len(files)} documents\n\n"
        summary += "\n".join(results)
        
        return summary
        
    except Exception as e:
        error_msg = f"❌ Error reloading documents: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def list_documents() -> str:
    """
    List all indexed documents with chunk counts.
    
    Returns:
        JSON-formatted list of documents
    """
    try:
        # Get all unique documents
        results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000  # Adjust if you have more documents
        )
        
        # Aggregate by document_id
        docs = {}
        for point in results[0]:
            payload = point.payload
            doc_id = payload['document_id']
            
            if doc_id not in docs:
                docs[doc_id] = {
                    "filename": payload['filename'],
                    "chunks": payload['total_chunks'],
                    "uploaded_at": payload['uploaded_at']
                }
        
        if not docs:
            return "No documents indexed yet."
        
        output = f"Found {len(docs)} indexed documents:\n\n"
        for doc in docs.values():
            output += f"• {doc['filename']} ({doc['chunks']} chunks)\n"
            output += f"  Uploaded: {doc['uploaded_at']}\n"
        
        return output
        
    except Exception as e:
        error_msg = f"❌ Error listing documents: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Initialize collection on startup
try:
    initialize_collection()
    logger.info("🚀 RAG MCP Server initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize: {e}")
