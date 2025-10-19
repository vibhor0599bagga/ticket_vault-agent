import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def load_data():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]

    docs = []
    for event in collection.find():
        content = (
            f"Event: {event['title']}\n"
            f"Location: {event['location']}\n"
            f"Venue: {event['venue']}\n"
            f"Date: {event['date']}\n"
            f"Time: {event['time']}\n"
            f"Price: ${event['price']}\n"
            f"Category: {event['category']}\n"
            f"Rating: {event['rating']}\n"
            f"Description: {event['description']}\n"
            f"Available Tickets: {event['availableTickets']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "price": event["price"],
                "location": event["location"],
                "venue": event["venue"],
                "category": event["category"],
                "date": event["date"],
                "rating": event["rating"]
            }
        ))

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="./event_vectors"
    )
    print("✅ MongoDB data loaded and embedded into Chroma.")

if __name__ == "__main__":
    load_data()