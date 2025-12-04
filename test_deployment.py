"""
Test script to verify the deployment works correctly
Run this after deploying to Render
"""
import requests
import json

# Replace with your actual Render URL
RENDER_URL = "https://your-app-name.onrender.com"
# For local testing, use:
# RENDER_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("=" * 50)
    print("Testing /api/health endpoint...")
    try:
        response = requests.get(f"{RENDER_URL}/api/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_debug():
    """Test debug endpoint to check configuration"""
    print("\n" + "=" * 50)
    print("Testing /api/debug endpoint...")
    try:
        response = requests.get(f"{RENDER_URL}/api/debug")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        data = response.json()
        if data.get("status") == "ok":
            print("\n✅ Configuration looks good!")
            print(f"   - Vector DB count: {data['vector_db']['count']}")
            print(f"   - MongoDB connected: {data['mongodb']['connected']}")
            print(f"   - MongoDB events: {data['mongodb']['count']}")
        else:
            print(f"\n⚠️ Configuration issue detected!")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat(query):
    """Test chat endpoint"""
    print("\n" + "=" * 50)
    print(f"Testing /api/chat endpoint with query: '{query}'")
    try:
        payload = {"query": query}
        response = requests.post(f"{RENDER_URL}/api/chat", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nAnswer: {data['answer']}")
            print(f"Session ID: {data['session_id']}")
            print(f"Is Event Related: {data['is_event_related']}")
            print(f"Processing Time: {data['processing_time']:.2f}s")
            return data['session_id']
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_followup(query, session_id):
    """Test followup query with session"""
    print("\n" + "=" * 50)
    print(f"Testing followup query: '{query}'")
    try:
        payload = {"query": query, "session_id": session_id}
        response = requests.post(f"{RENDER_URL}/api/chat", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nAnswer: {data['answer']}")
            print(f"Session ID: {data['session_id']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎟️ TicketVault Deployment Test Script")
    print("=" * 50)
    print(f"Testing URL: {RENDER_URL}\n")
    
    # Test 1: Health check
    health_ok = test_health()
    
    # Test 2: Debug endpoint
    debug_ok = test_debug()
    
    # Test 3: Chat endpoint
    session_id = test_chat("What concerts are happening in Mumbai?")
    
    # Test 4: Followup query
    if session_id:
        test_followup("What about comedy shows?", session_id)
    
    # Test 5: General query (non-event)
    test_chat("What is the capital of India?")
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("\nResults:")
    print(f"  - Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"  - Debug Check: {'✅ PASS' if debug_ok else '❌ FAIL'}")
    print(f"  - Chat Test: {'✅ PASS' if session_id else '❌ FAIL'}")
