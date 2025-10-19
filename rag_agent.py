from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import TypedDict

# Define state schema
class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    answer: str
    is_event_related: bool
    regenerate: bool

# Load vector database
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./event_vectors", embedding_function=embeddings)

# Initialize Conversation Memory using ChatMessageHistory
chat_history = ChatMessageHistory()

# Define LLM
# llm = OllamaLLM(model="mistral")
llm = OllamaLLM(model="mistral", temperature=0.2)


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

# --- LangGraph Nodes ---

def retrieve(state):
    query = state["query"]
    
    # Get conversation history from ChatMessageHistory
    messages = chat_history.messages
    
    # Build context-aware classification prompt
    history_context = ""
    if messages:
        # Format chat history for context - take last 6 messages (3 exchanges)
        history_messages = []
        for msg in messages[-6:]:
            role = "User" if msg.type == "human" else "Bot"
            history_messages.append(f"{role}: {msg.content}")
        if history_messages:
            history_context = f"\nPrevious conversation:\n" + "\n".join(history_messages) + "\n"
    
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
    
    classification = llm.invoke(classification_prompt).strip().upper()
    
    # If not event-related, skip retrieval
    if classification == "NO":
        state["retrieved_docs"] = []
        state["filtered_docs"] = []
        state["is_event_related"] = False
        return state
    
    # Otherwise, proceed with retrieval
    state["retrieved_docs"] = vectorstore.similarity_search(query, k=5)
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
    
    # Build conversation context
    history_context = ""
    if messages:
        history_messages = []
        for msg in messages[-6:]:  # Last 3 exchanges
            role = "User" if msg.type == "human" else "Bot"
            history_messages.append(f"{role}: {msg.content}")
        if history_messages:
            history_context = "Previous conversation:\n" + "\n".join(history_messages) + "\n\n"
    
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
    state["answer"] = llm.invoke(prompt_text)
    return state

def self_refine(state):
    initial_answer = state["answer"]
    refine_prompt = f"""
You are an expert assistant. Refine the following answer for clarity, conciseness, and accuracy. 
If the answer is already clear and correct, you may keep it as is.
If the answer is incomplete, unclear, or not relevant to the user's query, respond ONLY with: REGENERATE

Answer to refine:
{initial_answer}
"""
    refined = llm.invoke(refine_prompt)
    if refined.strip() == "REGENERATE":
        # Mark for regeneration by setting a flag in state
        state["regenerate"] = True
        return state
    state["answer"] = refined
    state["regenerate"] = False
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
        
        result = chatbot.invoke({"query": query})
        
        answer = result["answer"]
        print("Bot:", answer, "\n")
        
        # Save to ChatMessageHistory
        chat_history.add_user_message(query)
        chat_history.add_ai_message(answer)
