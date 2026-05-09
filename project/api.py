import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.rag_system import RAGSystem

# --- LIFESPAN MANAGEMENT ---
# This ensures heavy models load properly without blocking the port bind
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs ON STARTUP
    print("🚀 Initializing Agentic RAG System...")
    try:
        # We attach 'rag' to app.state so it's accessible in routes
        app.state.rag = RAGSystem()
        app.state.rag.initialize()
        print("✅ RAG System Ready!")
    except Exception as e:
        print(f"❌ Critical Initialization Error: {e}")
    
    yield
    # This runs ON SHUTDOWN
    print("Shutting down...")

# 1. Initialize FastAPI app with Lifespan
app = FastAPI(
    title="Agentic RAG Legal API",
    description="Backend API for Indian Legal Document Retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# 2. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Data Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# 4. Create the Health Check Route
@app.get("/health")
async def health_check():
    # Check if the RAG system is actually loaded yet
    if hasattr(app.state, "rag"):
        return {"status": "active", "database": app.state.rag.collection_name}
    return {"status": "starting", "message": "Models are still loading..."}

# 5. Create the Main Interaction Route
@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    if not hasattr(app.state, "rag"):
        raise HTTPException(status_code=503, detail="System is still initializing. Please try again in a minute.")
    
    try:
        rag = app.state.rag
        config = rag.get_config()
        
        # Format the input for LangGraph
        inputs = {"messages": [("user", request.query)]}
        
        # Invoke the agentic loop
        response = rag.agent_graph.invoke(inputs, config=config)
        
        # Extract final textual answer
        final_answer = response["messages"][-1].content
        
        return QueryResponse(answer=final_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
