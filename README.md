# 🤖 JarVish - Your LLM-Powered Personal Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-agent AI assistant powered by LangGraph with RAG memory, MCP tools, and seamless Google Workspace integration.

![JarVish Architecture](docs/architecture.png)

## ✨ Features

- 🧠 **Multi-Agent Orchestration** - Specialized agents for planning, execution, and synthesis
- 📚 **RAG Memory** - Semantic search through your documents using Qdrant vector database
- 🔗 **MCP Tools** - Standardized tool integration via Model Context Protocol
- 📧 **Gmail Integration** - Send, read, and search emails
- 📅 **Calendar Management** - Schedule and view events
- 🎯 **Graph-Based Reasoning** - LangGraph workflows for complex task decomposition
- 🐳 **Docker-Ready** - Containerized services for easy deployment

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key
- Google Workspace credentials (for Gmail/Calendar)

### 1. Clone the Repository

```bash
git clone https://github.com/vishwa419/JarVish.git
cd JarVish
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Server Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=jarvis_documents

# RAG Settings
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RAG_TOP_K=3
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Start Docker Services

Start Qdrant and RAG MCP Server:

```bash
docker-compose up -d
```

Verify services are running:

```bash
docker-compose ps
```

You should see:
- `jarvis-qdrant` (ports 6333, 6334)
- `jarvis-rag-mcp` (port 8001)

### 5. Setup Google Workspace MCP (Optional)

For Gmail and Calendar integration, install the Google Workspace MCP server:

```bash
# Clone the Google Workspace MCP repository
git clone https://github.com/taylorwilsdon/google_workspace_mcp.git mcp_servers/google_workspace

cd mcp_servers/google_workspace

# Install with Docker
docker build -t google-workspace-mcp .

# Run the container
docker run -d \
  --name jarvis-google-mcp \
  --network jarvis_jarvis-network \
  -p 8002:8002 \
  -v $(pwd)/credentials.json:/app/credentials.json \
  -v $(pwd)/token.json:/app/token.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  google-workspace-mcp
```

**Google Credentials Setup:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API and Google Calendar API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials as `credentials.json`
6. Place in `mcp_servers/google_workspace/`
7. Run authentication flow (first time only):

```bash
cd mcp_servers/google_workspace
python auth_setup.py  # Follow prompts to authenticate
```

### 6. Run JarVish

Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

JarVish is now running at: **http://localhost:8000**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│                 (FastAPI + HTMX)                         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              JarVish Multi-Agent Core                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Planner  │→ │ Resolver │→ │ Executor │→ │  Agg    │ │
│  │  Agent   │  │          │  │  (Loop)  │  │ regator │ │
│  └──────────┘  └──────────┘  └────┬─────┘  └─────────┘ │
│                                    │                     │
│                          ┌─────────▼─────────┐          │
│                          │  Tool Agents      │          │
│                          │  (Specialized)    │          │
│                          └─────────┬─────────┘          │
└────────────────────────────────────┼─────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────┐
        │                            │                        │
┌───────▼────────┐        ┌──────────▼──────────┐   ┌───────▼────────┐
│  Qdrant        │        │  Google Workspace    │   │  Other MCP     │
│  Vector DB     │        │  MCP Server          │   │  Tools         │
│  (Docker)      │        │  - Gmail             │   │                │
│                │        │  - Calendar          │   │                │
└────────────────┘        └──────────────────────┘   └────────────────┘
```

### Key Components

1. **Planner Agent** - Decomposes user requests into subtasks
2. **Dependency Resolver** - Creates execution order based on dependencies
3. **Executor** - Self-looping node that spawns tool-specific agents
4. **Tool Agents** - Specialized ReAct agents for each MCP tool
5. **Aggregator** - Synthesizes results into natural language

---

## 📁 Project Structure

```
JarVish/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── chain.py             # LangGraph multi-agent chain
│   ├── models.py            # Pydantic models & state
│   ├── config.py            # Configuration management
│   ├── utils.py             # Helper functions
│   └── mcp_client.py        # MCP client for tool discovery
├── mcp_servers/
│   ├── rag_server/          # RAG MCP server
│   │   ├── server.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── google_workspace/    # Google Workspace MCP (external)
│       ├── gmail_tool.py
│       ├── calendar_tool.py
│       └── server.py
├── frontend/
│   └── templates/
│       └── chat.html        # HTMX chat interface
├── data/
│   ├── qdrant/              # Vector database storage
│   └── uploads/             # Document uploads
├── tests/
│   └── test_agents.py       # Unit tests
├── docker-compose.yml       # Docker services
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md
```

---

## 🛠️ Usage Examples

### Basic Chat

```
You: "Hello JarVish!"
JarVish: "Hello! I'm JarVish, your AI assistant. I can help you search documents, 
          send emails, manage your calendar, and more. What can I do for you?"
```

### Document Search (RAG)

```
You: "What did Alice say about the project timeline?"
JarVish: *searches vector database*
         "According to your email from June 12th, Alice mentioned that the 
          project timeline has been extended to Q4 due to additional requirements."
```

### Email + Calendar Integration

```
You: "Find Bob's email address and schedule a meeting with him tomorrow at 3pm"
JarVish: *Step 1: Searches documents for Bob*
         *Step 2: Schedules calendar event*
         *Step 3: Sends meeting invite*
         "I found Bob Smith (bob@company.com) and scheduled a meeting for 
          tomorrow at 3pm. Calendar invite sent!"
```

### Complex Multi-Step Task

```
You: "Search my emails for project updates from last week, summarize them, 
      and create a calendar reminder to review them tomorrow"
JarVish: *Step 1: Gmail search*
         *Step 2: Summarize with LLM*
         *Step 3: Create calendar event*
         "I found 5 project update emails from last week. Summary: [...]
          I've created a reminder for tomorrow at 10am to review these updates."
```

---

## 🔧 Development

### Running Tests

```bash
pytest tests/ -v
```

### Debug Mode

Enable verbose logging in `.env`:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

Watch agent reasoning in terminal:

```bash
uvicorn backend.main:app --reload --log-level debug
```

### Adding New MCP Tools

1. Create tool in `mcp_servers/your_tool/`
2. Implement FastMCP server with `@mcp.tool()` decorator
3. Add tool discovery in `backend/utils.py`
4. Tools are auto-discovered by JarVish!

Example:

```python
# mcp_servers/your_tool/server.py
from fastmcp import FastMCP

mcp = FastMCP("YourTool")

@mcp.tool()
def your_function(param: str) -> str:
    """
    Description of what this tool does.
    The LLM reads this to decide when to use it.
    """
    # Your implementation
    return result

if __name__ == "__main__":
    mcp.run()
```

---

## 🐳 Docker Management

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f qdrant
docker-compose logs -f rag-mcp
```

### Restart Services

```bash
docker-compose restart
```

### Stop Services

```bash
docker-compose down
```

### Clean Everything (including volumes)

```bash
docker-compose down -v
```

---

## 📊 Monitoring

### Qdrant Dashboard

Access Qdrant's web UI at: http://localhost:6333/dashboard

- View collections
- Inspect vectors
- Monitor performance

### API Health Checks

```bash
# JarVish backend
curl http://localhost:8000/health

# Qdrant
curl http://localhost:6333/

# RAG MCP
curl http://localhost:8001/health
```

---

## 🚧 Troubleshooting

### Qdrant Connection Failed

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant

# Check logs
docker-compose logs qdrant
```

### Google Workspace Authentication Issues

```bash
# Re-run authentication
cd mcp_servers/google_workspace
python auth_setup.py

# Verify credentials
ls -la credentials.json token.json
```

### OpenAI API Errors

- Verify API key in `.env`
- Check API quota: https://platform.openai.com/usage
- Ensure `OPENAI_API_KEY` starts with `sk-`

### Port Already in Use

```bash
# Find process using port
lsof -i :8000  # or 6333, 8001

# Kill process
kill -9 <PID>

# Or change port in .env
BACKEND_PORT=8001
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Resources

- **LangChain Documentation**: https://python.langchain.com/
- **LangGraph Guide**: https://langchain-ai.github.io/langgraph/
- **FastMCP Protocol**: https://github.com/jlowin/fastmcp
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Google Workspace MCP**: https://github.com/taylorwilsdon/google_workspace_mcp

---

## 🌟 Acknowledgments

Special thanks to:
- [Taylor Wilsdon](https://github.com/taylorwilsdon) for the excellent [Google Workspace MCP implementation](https://github.com/taylorwilsdon/google_workspace_mcp)
- The LangChain and LangGraph teams for building amazing agentic frameworks
- The Qdrant team for a blazing-fast vector database

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔮 Future Roadmap

- [ ] Persistent short-term memory storage
- [ ] Google Drive integration with automatic RAG
- [ ] LLMOps observability dashboard
- [ ] Voice interface
- [ ] Multi-user sessions with authentication
- [ ] Mobile app
- [ ] Function calling optimization
- [ ] Cost tracking and analytics

---

## 💬 Support

For questions or issues:
- Open an [Issue](https://github.com/vishwa419/JarVish/issues)
- Discussions: [GitHub Discussions](https://github.com/vishwa419/JarVish/discussions)

---

**Built with ❤️ by [Vishwa](https://github.com/vishwa419)**

*The future is conversational. The future is JarVish.* 🚀
