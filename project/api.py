from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.rag_system import RAGSystem

# 1. Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG Legal API",
    description="Backend API for Indian Legal Document Retrieval",
    version="1.0.0"
)

# 2. Configure CORS (Crucial for connecting a frontend website later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize the LangGraph RAG System globally
print("Initializing RAG System...")
rag = RAGSystem()
rag.initialize()
print("RAG System Ready!")

# 4. Define Data Models for Input and Output
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# 5. Create the Health Check Route
@app.get("/health")
async def health_check():
    return {"status": "active", "database": rag.collection_name}

# 6. Create the Main Interaction Route
@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    try:
        # Fetch the LangGraph config (which includes your thread_id)
        config = rag.get_config()
        
        # Format the input for LangGraph
        inputs = {"messages": [("user", request.query)]}
        
        # Invoke the agentic loop
        response = rag.agent_graph.invoke(inputs, config=config)
        
        # Extract the final textual answer from the last message in the graph state
        final_answer = response["messages"][-1].content
        
        return QueryResponse(answer=final_answer)

    except Exception as e:
        # Catch errors (like rate limits) and return a clean HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))