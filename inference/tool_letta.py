import os
import json
import datetime
from typing import Union, Dict, List, Optional
from qwen_agent.tools import BaseTool
from letta_client import Letta, LettaError

class LettaTool(BaseTool):
    name = 'letta'
    description = 'Interact with a Letta (MemGPT) agent for long-term memory and state management.'
    parameters = [{
        'name': 'message',
        'type': 'string',
        'description': 'The message to send to the Letta agent.',
        'required': True
    }, {
        'name': 'agent_id',
        'type': 'string',
        'description': 'The ID or name of the Letta agent to interact with. If not provided, uses a default agent.',
        'required': False
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.api_key = os.getenv("LETTA_API_KEY")
        self.base_url = os.getenv("LETTA_BASE_URL", "https://api.letta.com") # Default to cloud
        self.client = None
        self.default_agent_id = os.getenv("LETTA_AGENT_ID")
        
        if self.api_key:
            try:
                self.client = Letta(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                print(f"Failed to initialize Letta client: {e}")

    def _ensure_agent(self):
        """Ensures a default agent is selected or created."""
        if self.default_agent_id:
            return

        if not self.client:
            return

        try:
            # Check for ephemeral mode
            is_ephemeral = os.getenv("LETTA_EPHEMERAL", "true").lower() == "true"
            print(f"[Letta] Ephemeral mode: {'ENABLED - Starting with clean memory' if is_ephemeral else 'DISABLED - Reusing existing agent'}")
            
            found_agent = None
            if not is_ephemeral:
                agents_page = self.client.agents.list()
                # Handle SyncArrayPage which might not be subscriptable
                for agent in agents_page:
                    found_agent = agent
                    break
            
            if found_agent:
                self.default_agent_id = found_agent.id
                print(f"[Letta] Using existing agent: {found_agent.name} ({found_agent.id})")
            else:
                # Create a new agent
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                agent_name = f"DeepResearchAgent-{timestamp}" if is_ephemeral else "DeepResearchAgent"
                
                print(f"[Letta] Creating new agent: {agent_name}")
                agent = self.client.agents.create(name=agent_name)
                self.default_agent_id = agent.id
                print(f"[Letta] Agent created with ID: {self.default_agent_id}")
        except Exception as e:
            print(f"Error ensuring agent: {e}")

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not self.client:
            return "Error: Letta client not initialized. Please set LETTA_API_KEY."

        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"message": params}

        message = params.get('message')
        
        # Ensure agent is initialized before checking params
        self._ensure_agent()
        
        agent_id = params.get('agent_id') or self.default_agent_id

        if not message:
            return "Error: 'message' parameter is required."

        if not agent_id:
            return "Error: No agent found or created."

        try:
            response = self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": message}]
            )
            
            # Extract the response content
            # Simplified extraction to avoid issues with message history pagination
            output = []
            
            # Try to get messages from response
            if hasattr(response, 'messages'):
                messages = response.messages
                for msg in messages:
                    if hasattr(msg, 'message_type') and msg.message_type == 'assistant_message':
                        output.append(msg.content)
                    elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                        output.append(msg.get('content', ''))
            
            # If we got output, return it
            if output:
                return "\n".join(output)
            
            # Otherwise, try to convert response to string
            return str(response) if response else "No response from Letta agent."

        except LettaError as e:
            error_msg = str(e)
            # Check if this is the message index error
            if "No assistant message found from indices" in error_msg:
                print(f"[Letta] Warning: Message history pagination error, creating fresh agent")
                # Try to create a new agent as a fallback
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    new_agent = self.client.agents.create(name=f"DeepResearchAgent-{timestamp}")
                    self.default_agent_id = new_agent.id
                    print(f"[Letta] Created fresh agent: {self.default_agent_id}")
                    # Retry the call with the new agent
                    response = self.client.agents.messages.create(
                        agent_id=self.default_agent_id,
                        messages=[{"role": "user", "content": message}]
                    )
                    return str(response) if response else "No response from Letta agent."
                except Exception as retry_error:
                    return f"Letta API Error (retry failed): {retry_error}"
            return f"Letta API Error: {e}"
        except Exception as e:
            return f"Unexpected error calling Letta: {e}"

    def query_memory(self, query: str) -> Optional[str]:
        """Queries Letta memory for information. Returns None if no relevant info found."""
        prompt = f"Do you have any information in your memory regarding: '{query}'? If yes, please provide it concisely. If no, strictly reply with 'NO_INFO'."
        response = self.call({"message": prompt})
        if "NO_INFO" in response:
            return None
        return response

    def save_memory(self, content: str):
        """Saves content to Letta memory."""
        prompt = f"Please save the following tool output to your memory for future reference:\n{content}"
        self.call({"message": prompt})

    def get_core_memory(self) -> str:
        """Retrieves the agent's core memory block."""
        if not self.client:
             return "Letta client not initialized."
        
        self._ensure_agent()
        if not self.default_agent_id:
            return "Letta agent not initialized."

        try:
            memory = self.client.agents.get_core_memory(agent_id=self.default_agent_id)
            return str(memory)
        except Exception as e:
            return f"Error retrieving core memory: {e}"

    def list_messages(self, limit: int = 10) -> str:
        """Retrieves recent messages from the agent's history."""
        if not self.client:
             return "Letta client not initialized."
        
        self._ensure_agent()
        if not self.default_agent_id:
            return "Letta agent not initialized."

        try:
            messages = self.client.agents.messages.list(agent_id=self.default_agent_id, limit=limit)
            # Format messages
            output = []
            for msg in messages:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', '')
                if hasattr(msg, 'message_type'):
                     role = f"{role} ({msg.message_type})"
                output.append(f"[{role}]: {content}")
            return "\n".join(output)
        except Exception as e:
            return f"Error retrieving messages: {e}"
