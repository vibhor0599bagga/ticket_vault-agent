from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import TypedDict
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# Define state schema
class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    answer: str
    is_event_related: bool
    regenerate: bool
    attempts: int

# Load vector database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # 768-dimensional embeddings
)
vectorstore = Chroma(persist_directory="./event_vectors", embedding_function=embeddings)

# Initialize Conversation Memory using ChatMessageHistory
chat_history = ChatMessageHistory()

# Define LLMs using Groq (much faster and hostable)
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast and efficient
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# Separate LLM for follow-up detection with slightly higher temperature
llm_followup = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)


prompt = ChatPromptTemplate.from_template("""
You are a helpful event assistant for TicketVault (a customer-to-customer ticket selling platform).

The user asked: {query}
Here is event info you found:
{context}

IMPORTANT:
- Only list events that directly match the user's query. For example, if the user asks about "Avengers", only show events with "Avengers" in their name or description.
- Do NOT show unrelated events, even if they are similar.
- If non related question is asked, reply: "I can only help with event-related queries."
- If the question is not related to events, tickets, or prices, reply: "I can only help with event-related queries."

Respond with clear bullet points showing event name, date, location, and price.
Respond to the question in concise manner and be specific to the question asked.
""")

# ---------- Follow-up detector (use history only when needed) ----------
def is_followup(query: str) -> bool:
    """
    Returns True if the new user message depends on prior turns
    (e.g., 'what about cheaper ones?', 'and the venue?', 'those tickets'), else False.
    """
    msgs = chat_history.messages[-6:]  # last ~3 exchanges
    if not msgs:
        return False
    convo = []
    for m in msgs:
        role = "User" if m.type == "human" else "Bot"
        convo.append(f"{role}: {m.content}")
    history_text = "\n".join(convo)
    prompt = f"""
Decide if the NEW user message depends on the previous conversation (is a follow-up),
or if it starts a new, independent topic.

Previous conversation:
{history_text}

New message: "{query}"

Reply with ONLY YES or NO.
"""
    # Use separate LLM instance for follow-up detection
    response = llm_followup.invoke(prompt)
    out = response.content.strip().upper()
    return out == "YES"

# --- LangGraph Nodes ---

def retrieve(state):
    query = state["query"]
    # Initialize attempts counter if missing
    if state.get("attempts") is None:
        state["attempts"] = 0
    
    # Use history only if this is a follow-up
    history_context = ""
    if is_followup(query):
        msgs = chat_history.messages[-6:]
        if msgs:
            history_lines = []
            for msg in msgs:
                role = "User" if msg.type == "human" else "Bot"
                history_lines.append(f"{role}: {msg.content}")
            history_context = f"\nPrevious conversation:\n" + "\n".join(history_lines) + "\n"
 
    # Step 1: Use LLM to check if query is event-related
    classification_prompt = f"""
You are a query classifier for an event ticketing platform called TicketVault.
{history_context}
Current user query: "{query}"

Is this query related to events, tickets, concerts, shows, movies, sports, venues, prices, or booking?
Consider the conversation context above when making your decision.
Respond with ONLY one word: "YES" or "NO"

Examples:
- "what is the price of taylor swift tickets" -> YES
- "show me concerts in New York" -> YES
- "what is the capital of France" -> NO
- "who is the president" -> NO
- "I want to buy avengers ticket" -> YES
"""
    
    # Extract content from AIMessage object returned by Groq
    response = llm.invoke(classification_prompt)
    classification = response.content.strip().upper()
    
    # If not event-related, skip retrieval
    if classification == "NO":
        state["retrieved_docs"] = []
        state["is_event_related"] = False
        return state
    
    # Otherwise, proceed with retrieval (keep it tight)
    state["retrieved_docs"] = vectorstore.similarity_search(query, k=3)
    state["is_event_related"] = True
    return state

def generate(state):
    # Check if query was classified as non-event-related
    if not state.get("is_event_related", True):
        state["answer"] = "I can only help with event-related queries such as finding tickets, checking prices, venues, dates, or booking events."
        return state
    
    docs = state.get("retrieved_docs", [])
    
    # Get conversation history from ChatMessageHistory
    messages = chat_history.messages
    
    # If no documents found after filtering
    if not docs:
        state["answer"] = "No matching events found. Please try a different search term or check back later for new events."
        return state
    
    # Build conversation context (only if this is a follow-up)
    history_context = ""
    if is_followup(state["query"]):
        msgs = chat_history.messages[-6:]
        if msgs:
            lines = []
            for msg in msgs:
                role = "User" if msg.type == "human" else "Bot"
                lines.append(f"{role}: {msg.content}")
            history_context = "Previous conversation:\n" + "\n".join(lines) + "\n\n"
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Enhanced prompt with conversation history and budget handling
    enhanced_prompt = ChatPromptTemplate.from_template("""
You are a helpful event assistant for TicketVault (a customer-to-customer ticket selling platform).

{history}Current user query: {query}

Here is event info you found:
{context}

IMPORTANT:
- Consider the conversation history when answering.
- If the user mentions a budget (e.g., "under $50", "below $100"), filter and show only events within that budget.
- Only list events that directly match the user's query and any constraints (budget, location, date, etc.).
- IMPORTANT: DO NOT show unrelated events, even if they are similar.
- If non-related question is asked, reply: "I can only help with event-related queries."
- If the question is not related to events, tickets, or prices, reply: "I can only help with event-related queries."

Respond with clear bullet points showing event name, date, location, and price.
Be conversational and helpful. Reference previous context when relevant.
""")
    
    prompt_text = enhanced_prompt.format(
        history=history_context,
        query=state["query"], 
        context=context
    )
    response = llm.invoke(prompt_text)
    state["answer"] = response.content
    return state

def self_refine(state):
    # Cap regeneration attempts to prevent infinite loops
    max_attempts = 2
    if state.get("attempts", 0) >= max_attempts:
        state["regenerate"] = False
        # Provide a safe fallback if we still don't have a grounded answer
        if not state.get("answer"):
            state["answer"] = "I don't have enough event data to answer that. Please try a different query."
        return state

    # If out-of-scope or no context, keep the safe answer and stop here
    if not state.get("is_event_related", True):
        state["regenerate"] = False
        return state

    docs = state.get("retrieved_docs", [])
    if not docs:
        state["regenerate"] = False
        return state

    initial_answer = state["answer"]

    # First, attempt a light refinement
    refine_prompt = f"""
You are an expert editor. Refine the following answer for clarity, conciseness, and correctness.
Do not add new facts that are not already present in the answer.
If the answer is incomplete, unclear, or likely not answering the user's question, respond ONLY with: REGENERATE

Answer to refine:
{initial_answer}
"""
    response = llm.invoke(refine_prompt)
    refined = response.content.strip()

    # If the refiner explicitly asked to regenerate, set the flag and return
    if refined == "REGENERATE":
        state["attempts"] = state.get("attempts", 0) + 1
        state["regenerate"] = True
        return state

    # Groundedness judge: verify the refined answer is supported by the retrieved context
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""

    judge_prompt = f"""
You are a strict verifier. Determine if the proposed answer is FULLY supported by the provided context for the given user query.
If ANY part of the answer is not explicitly supported by the context, respond ONLY with: REGENERATE
Otherwise, respond ONLY with: OK

User query:
{state['query']}

Context:
{context}

Proposed answer:
{refined}
"""
    response = llm.invoke(judge_prompt)
    verdict = response.content.strip().upper()

    if verdict == "OK":
        state["answer"] = refined
        state["regenerate"] = False
        return state

    # Not grounded -> trigger regeneration cycle
    state["answer"] = refined
    state["attempts"] = state.get("attempts", 0) + 1
    state["regenerate"] = True
    return state

def should_regenerate(state):
    """Route to generator if regeneration needed, otherwise end."""
    if state.get("regenerate", False):
        return "generator"
    return END

# --- Build the graph ---

graph = StateGraph(AgentState)
graph.add_node("retriever", retrieve)
graph.add_node("generator", generate)
graph.add_node("self_refine", self_refine)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", "self_refine")
graph.add_conditional_edges("self_refine", should_regenerate)

chatbot = graph.compile()

# --- Run Interactively ---
if __name__ == "__main__":
    print("🎟️ TicketVault Chatbot — Ask about events, budgets, or ticket prices.\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        if query.lower() in ["reset", "clear", "new topic"]:
            # Clear memory to avoid leakage across topics
            chat_history.clear()
            print("🔁 Conversation history cleared.")
            continue
        
        result = chatbot.invoke({"query": query})
        
        answer = result["answer"]
        print("Bot:", answer, "\n")
        
        # Save to ChatMessageHistory
        chat_history.add_user_message(query)
        chat_history.add_ai_message(answer)
        # Reset attempts between user turns
        if "attempts" in result:
            result["attempts"] = 0
