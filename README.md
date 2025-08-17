"# Legal Document Analyzer - Dual Processing Architecture

🏛️ **AI-Powered Legal Document Analysis with Dual Processing Modes**

A sophisticated legal document analysis system offering two complementary processing approaches:

- **Direct Processing**: Fast AI analysis (<20s) for immediate insights
- **Vector Processing**: Document storage and semantic search for long-term knowledge management

---

## 🚀 Features

### **Direct Processing (Fast Analysis)**

- ⚡ **Sub-20 second response** times with parallel AI agents
- 📄 **Direct PDF processing** with Google Gemini
- 🔍 **Comprehensive analysis**: Risk assessment, key highlights, obligations
- 🎯 **Master-sub agentic architecture** for optimal performance
- 📊 **Confidence metrics** and recommendations

### **Vector Processing (Knowledge Base)**

- 📝 **Intelligent text chunking** for optimal semantic search
- 🧠 **Google Embedding API** integration (text-embedding-004)
- 🗄️ **Redis vector database** with enterprise-grade search
- 🔍 **Semantic similarity search** across document corpus
- 🤖 **RAG-ready infrastructure** for future Q&A capabilities

---

## 🏗️ Architecture

```
📄 PDF Upload
     ↓
┌─────────────────┐
│   File Router   │
│    (FastAPI)    │
└─────────────────┘
     ↓
┌─────────────────┴─────────────────┐
│                                   │
▼                                   ▼
🚀 Direct Processing              📚 Vector Processing
┌─────────────────┐              ┌─────────────────┐
│ Master Agent    │              │ Text Extract    │
│ 4 Parallel AI   │              │ Chunk & Embed   │
│ Agents (<20s)   │              │ Redis Storage   │
└─────────────────┘              │ Semantic Search │
     ↓                           └─────────────────┘
📊 Instant Analysis                    ↓
                                 🔍 Searchable Knowledge
```

### **Technology Stack**

- **Backend**: FastAPI + Python 3.8+
- **AI Models**: Google Gemini 2.5 Flash + text-embedding-004
- **Workflow**: LangGraph for agent orchestration
- **Vector DB**: Redis Stack with RediSearch
- **Document Processing**: PyMuPDF + PaddleOCR
- **Embedding**: LangChain + Google Embeddings

---

## 🛠️ Quick Setup

### **1. Prerequisites**

```bash
# Python 3.8+
python --version

# Docker (optional, for Redis)
docker --version
docker-compose --version
```

### **2. Automated Setup**

```bash
# Clone and navigate to project
cd Code_strom

# Run setup script
python setup_dual_processing.py
```

### **3. Manual Setup**

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env.example .env
# Edit .env with your Google API key

# Start Redis (with Docker)
docker-compose up -d

# Start application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 📋 Environment Configuration

Edit `.env` file with your configurations:

```bash
# Required: Google API Key
GOOGLE_API_KEY=your_google_api_key_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=yourpassword

# Optional: Performance tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_WORKERS=4
```

---

## 🔗 API Endpoints

### **Processing Endpoints**

```bash
# Fast AI Analysis (Direct)
POST /process_direct
- Input: PDF file (max 10MB)
- Output: Comprehensive analysis in <20s
- Use: Immediate document insights

# Vector Storage (Knowledge Base)
POST /process_vector
- Input: PDF file (max 10MB)
- Output: Document stored in vector DB
- Use: Building searchable document corpus

# Semantic Search
POST /search_documents
- Input: Query string
- Output: Similar document chunks
- Use: Find relevant content across documents
```

### **Utility Endpoints**

```bash
GET /health           # Service health check
GET /vector_stats     # Vector database statistics
GET /docs            # Interactive API documentation
```

---

## 🧪 Testing the System

### **1. Test Direct Processing**

```bash
curl -X POST "http://localhost:8000/process_direct" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### **2. Test Vector Processing**

```bash
# Store document
curl -X POST "http://localhost:8000/process_vector" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"

# Search documents
curl -X POST "http://localhost:8000/search_documents" \
  -H "Content-Type: application/json" \
  -d '{"query": "termination clause", "top_k": 5}'
```

### **3. Check System Health**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/vector_stats
```

---

## 📊 Performance Benchmarks

### **Direct Processing**

- ✅ **Target**: <20 seconds total processing
- ⚡ **Parallel execution** with 4 concurrent AI agents
- 🎯 **Optimized** for immediate results

### **Vector Processing**

- 📝 **Text extraction**: ~2-5 seconds
- 🧠 **Embedding generation**: ~5-15 seconds
- 💾 **Vector storage**: ~1-3 seconds
- 🔍 **Search queries**: <1 second

---

## 🔧 Advanced Configuration

### **Redis Tuning**

Edit `redis.conf` for production:

```bash
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
appendonly yes
```

### **Chunking Optimization**

Adjust in `.env`:

```bash
CHUNK_SIZE=1500      # Larger for complex documents
CHUNK_OVERLAP=300    # More overlap for better context
```

### **Performance Scaling**

```bash
MAX_WORKERS=8        # More workers for heavy loads
EMBEDDING_BATCH_SIZE=20  # Larger batches for efficiency
```

---

## 🐳 Docker Deployment

### **Full Stack Deployment**

```bash
# Start Redis
docker-compose up -d

# Build application image
docker build -t legal-analyzer .

# Run application
docker run -p 8000:8000 --env-file .env legal-analyzer
```

---

## 📈 Monitoring & Maintenance

### **Health Monitoring**

- Monitor `/health` endpoint
- Check Redis connection status
- Track processing times and success rates

### **Vector Database Maintenance**

- Monitor storage usage with `/vector_stats`
- Regular backup of Redis data
- Cleanup old or irrelevant documents

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with clear description

---

## 📄 License

This project is licensed under the MIT License.

---

## 🆘 Troubleshooting

### **Common Issues**

**Redis Connection Failed**

```bash
# Check Redis status
docker ps | grep redis
docker logs legal-analyzer-redis

# Restart Redis
docker-compose restart
```

**API Key Issues**

```bash
# Verify API key in .env
grep GOOGLE_API_KEY .env

# Test API access
curl "https://generativelanguage.googleapis.com/v1/models?key=YOUR_KEY"
```

**Performance Issues**

```bash
# Check system resources
htop

# Monitor application logs
tail -f app.log

# Reduce concurrent workers if needed
export MAX_WORKERS=2
```

---

**🎉 Ready to analyze legal documents with dual processing power!**"
