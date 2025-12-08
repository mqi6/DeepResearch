import os
from letta_client import Letta
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LETTA_API_KEY")
base_url = os.getenv("LETTA_BASE_URL", "https://api.letta.com")

if not api_key:
    print("LETTA_API_KEY not set.")
    exit(1)

try:
    client = Letta(api_key=api_key, base_url=base_url)
    print("Client initialized.")
    
    # List agents to get an ID
    agents_page = client.agents.list()
    agent_id = None
    for agent in agents_page:
        agent_id = agent.id
        break
    
    if not agent_id:
        print("No agents found.")
        exit(0)
        
    print(f"Agent ID: {agent_id}")
    
    # Try to get agent details
    print("\n--- Agent Memory Inspection ---")
    try:
        agent = client.agents.retrieve(agent_id=agent_id)
        if hasattr(agent, 'memory'):
            print(f"Agent Memory Type: {type(agent.memory)}")
            print(f"Agent Memory Dir: {dir(agent.memory)}")
            print(f"Agent Memory Vars: {vars(agent.memory) if hasattr(agent.memory, '__dict__') else 'No __dict__'}")
        else:
            print("Agent has no 'memory' attribute.")
    except Exception as e:
        print(f"Error retrieving agent: {e}")

    # Inspect messages
    print("\n--- Message Inspection ---")
    try:
        messages = client.agents.messages.list(agent_id=agent_id, limit=5)
        for msg in messages:
            print(f"\nMessage Type: {type(msg)}")
            print(f"Message Dir: {dir(msg)}")
            if hasattr(msg, '__dict__'):
                print(f"Message Dict: {msg.__dict__}")
    except Exception as e:
        print(f"Error listing messages: {e}")

except Exception as e:
    print(f"Error: {e}")
