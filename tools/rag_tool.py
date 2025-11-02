"""
RAG (Retrieval Augmented Generation) Tool
Searches through documents using ChromaDB vector database.
"""
import os
from pathlib import Path
from typing import List, Optional
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

from backend.config import settings

logger = logging.getLogger(__name__)


class RAGTool:
    """
    Document search tool using ChromaDB vector database.
    
    This tool:
    1. Loads documents from data/docs/
    2. Splits them into chunks
    3. Creates embeddings using OpenAI
    4. Stores in ChromaDB
    5. Provides semantic search functionality
    """
    
    def __init__(self):
        """Initialize RAG tool with ChromaDB and embeddings."""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        
        self.docs_path = Path(settings.docs_path)
        self.chroma_path = Path(settings.chroma_db_path)
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vectorstore: Optional[Chroma] = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self) -> None:
        """
        Initialize or load ChromaDB vector store.
        If documents exist and DB is empty, load them.
        """
        try:
            # Create directories if they don't exist
            self.docs_path.mkdir(parents=True, exist_ok=True)
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing database
            self.vectorstore = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=self.embeddings,
                collection_name="jarvis_docs"
            )
            
            # Check if we need to load documents
            collection_count = self.vectorstore._collection.count()
            
            if collection_count == 0:
                logger.info("ChromaDB is empty, loading documents...")
                self.load_documents()
            else:
                logger.info(f"ChromaDB loaded with {collection_count} chunks")
                
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            raise
    
    def load_documents(self) -> int:
        """
        Load all documents from docs directory into ChromaDB.
        
        Returns:
            Number of document chunks loaded
        """
        try:
            # Check if docs directory has files
            doc_files = list(self.docs_path.glob("*.txt"))
            
            if not doc_files:
                logger.warning(f"No .txt files found in {self.docs_path}")
                return 0
            
            logger.info(f"Loading {len(doc_files)} document(s)...")
            
            # Load documents
            loader = DirectoryLoader(
                str(self.docs_path),
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents loaded")
                return 0
            
            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(splits)} chunks")
            
            # Add to vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_path),
                collection_name="jarvis_docs"
            )
            
            logger.info(f"✅ Successfully loaded {len(splits)} chunks into ChromaDB")
            return len(splits)
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return (default from settings)
            
        Returns:
            List of dicts with 'content' and 'metadata'
        """
        if not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return []
        
        try:
            k = top_k or settings.rag_top_k
            
            # Perform similarity search
            results = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def search_with_scores(self, query: str, top_k: Optional[int] = None) -> List[tuple]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            return []
        
        try:
            k = top_k or settings.rag_top_k
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted = []
            for doc, score in results:
                formatted.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error in scored search: {e}")
            return []
    
    def reload_documents(self) -> int:
        """
        Clear existing database and reload all documents.
        Useful when documents have been updated.
        
        Returns:
            Number of chunks loaded
        """
        try:
            # Delete existing collection
            if self.vectorstore:
                self.vectorstore.delete_collection()
            
            # Reinitialize
            self._initialize_vectorstore()
            
            return self.vectorstore._collection.count()
            
        except Exception as e:
            logger.error(f"Error reloading documents: {e}")
            raise
    
    def format_results_for_llm(self, results: List[dict]) -> str:
        """
        Format search results as context for LLM.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string for LLM context
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "unknown")
            content = result.get("content", "")
            
            context_parts.append(
                f"[Document {i} - {Path(source).name}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)


# Global RAG tool instance
rag_tool = RAGTool()


# MCP Tool Functions
def rag_search(query: str, top_k: int = 3) -> str:
    """
    Search through local documents for relevant information using semantic similarity.
    
    Use this tool when the user asks about:
    - Finding information in their documents
    - Looking up specific topics or people
    - Retrieving details from their knowledge base
    - Questions about project documentation
    
    Args:
        query: The search query or topic to find in documents
        top_k: Number of relevant document chunks to return (default: 3)
        
    Returns:
        Formatted text containing the most relevant document passages
        
    Example:
        rag_search("Alice Johnson") → Returns passages mentioning Alice
        rag_search("RAG implementation details") → Returns technical docs
    """
    try:
        results = rag_tool.search(query, top_k=top_k)
        
        if not results:
            return f"No documents found matching '{query}'. The knowledge base may be empty or the query is too specific."
        
        formatted = rag_tool.format_results_for_llm(results)
        return f"Found {len(results)} relevant document(s) for '{query}':\n\n{formatted}"
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Error searching documents: {str(e)}"


def reload_documents() -> str:
    """
    Reload all documents from the docs folder into the vector database.
    
    Use this when:
    - New documents have been added
    - Existing documents have been updated
    - The vector database needs to be refreshed
    
    Returns:
        Status message with number of chunks loaded
    """
    try:
        count = rag_tool.reload_documents()
        return f"✅ Successfully reloaded {count} document chunks from {settings.docs_path}"
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return f"❌ Error reloading documents: {str(e)}"
