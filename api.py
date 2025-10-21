from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import uuid
import time
from dotenv import load_dotenv

# Import your chatbot components
from rag_agent import chatbot, llm, chat_history, AgentState, is_followup as agent_is_followup
import rag_agent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="TicketVault Chatbot API",
    description="API for TicketVault event search and ticket assistance",
    version="1.0.0"
)

# Add CORS middleware to allow your Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store chat sessions (for multiple users)
chat_sessions = {}

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    is_event_related: bool
    processing_time: float
    
def get_session_history(session_id: str):
    """Get or create chat history for session"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = chat_history.__class__()
    return chat_sessions[session_id]

@app.post("/api/chat", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a chatbot query and return the response"""
    start_time = time.time()
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get session-specific chat history
    session_history = get_session_history(session_id)
    
    # Add explicit check for new sessions (first message)
    is_first_message = len(session_history.messages) == 0
    logger.info(f"Processing request for session {session_id}, history length: {len(session_history.messages)}")
    
    try:
        # Ensure complete isolation with deep copy of messages
        temp_chat_history = chat_history.__class__()
        
        if session_history.messages:
            # Copy messages to temporary history
            for msg in session_history.messages:
                if msg.type == "human":
                    temp_chat_history.add_user_message(msg.content)
                else:
                    temp_chat_history.add_ai_message(msg.content)
        
        # Save original history
        original_history = chat_history.messages.copy() if hasattr(chat_history, "messages") else []
        
        # Temporarily use session history
        if hasattr(chat_history, "messages"):
            chat_history.messages = temp_chat_history.messages
        
        # Force is_followup to return False for first messages
        if is_first_message:
            # Monkey patch for first message in session
            original_is_followup = rag_agent.is_followup
            rag_agent.is_followup = lambda x: False
            logger.info("First message in session - forcing is_followup=False")
        
        # Process the query
        result = chatbot.invoke({"query": request.query})
        
        # Restore original is_followup if needed
        if is_first_message:
            rag_agent.is_followup = original_is_followup
        
        # Get the answer
        answer = result.get("answer", "Sorry, I couldn't process your request.")
        is_event_related = result.get("is_event_related", False)
        
        # Add to session history
        session_history.add_user_message(request.query)
        session_history.add_ai_message(answer)
        
        # Background task to restore the original history
        def restore_history():
            if hasattr(chat_history, "messages"):
                chat_history.messages = original_history
                
        background_tasks.add_task(restore_history)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f}s")
        
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            is_event_related=is_event_related,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_session(request: Dict[str, str]):
    """Reset a chat session"""
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
        
    if session_id in chat_sessions:
        chat_sessions[session_id].clear()
        logger.info(f"Session {session_id} cleared")
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": llm.model}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting TicketVault API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)