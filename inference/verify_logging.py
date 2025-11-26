import os
import shutil
import time
import uuid
from unittest.mock import MagicMock
import sys
import json

# Mock missing dependencies
sys.modules["alibabacloud_docmind_api20220711"] = MagicMock()
sys.modules["alibabacloud_docmind_api20220711.client"] = MagicMock()
sys.modules["alibabacloud_tea_openapi"] = MagicMock()
sys.modules["alibabacloud_tea_openapi.models"] = MagicMock()
sys.modules["alibabacloud_tea_util"] = MagicMock()
sys.modules["alibabacloud_tea_util.client"] = MagicMock()
sys.modules["alibabacloud_credentials"] = MagicMock()
sys.modules["alibabacloud_credentials.client"] = MagicMock()
sys.modules["sandbox_fusion"] = MagicMock()
sys.modules["tabulate"] = MagicMock()

from react_agent import MultiTurnReactAgent

def mock_call_server(messages, planning_port=None, max_tries=10):
    last_msg = messages[-1]
    content = last_msg.get('content', '')
    
    print(f"DEBUG: Last msg role: {last_msg['role']}")
    print(f"DEBUG: Content: {content}")
    print(f"DEBUG: Messages: {messages}")

    if last_msg['role'] == 'user':
        # First turn: call search
        if "User:" in content or "Question:" in content or len(messages) <= 2:
             print("DEBUG: Deciding to call search")
             return '<tool_call>{"name": "search", "arguments": {"query": "test query"}}</tool_call>'
        
        # Second turn: call visit
        if "search" in str(messages):
             # Check if we haven't visited yet
             has_visit = any('visit' in m.get('content', '') and m.get('role') == 'assistant' for m in messages)
             print(f"DEBUG: Has visit: {has_visit}")
             if not has_visit:
                 print("DEBUG: Deciding to call visit")
                 return '<tool_call>{"name": "visit", "arguments": {"url": "http://example.com"}}</tool_call>'
             else:
                 print("DEBUG: Deciding to answer")
                 return '<answer>The answer is 42.</answer>'
                
    return "Unexpected state"

def test_logging():
    # Clean up logs directory
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    # Setup agent
    llm_cfg = {
        'model': 'mock-model',
        'generate_cfg': {
            'temperature': 0.0,
            'top_p': 1.0,
        },
        'model_type': 'qwen_dashscope'
    }
    
    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["search", "visit"]
    )
    
    # Mock call_server
    agent.call_server = mock_call_server
    
    # Mock count_tokens
    agent.count_tokens = MagicMock(return_value=10)
    agent.count_text_tokens = MagicMock(return_value=5)
    
    # Mock tools
    mock_search = MagicMock()
    mock_search.call.return_value = "Search Result"
    mock_search.name = "search"
    
    mock_visit = MagicMock()
    mock_visit.call.return_value = "Visit Result"
    mock_visit.name = "visit"
    
    # Patch TOOL_MAP
    import react_agent as react_agent_module
    react_agent_module.TOOL_MAP['search'] = mock_search
    react_agent_module.TOOL_MAP['visit'] = mock_visit

    # Run the agent
    data = {
        'item': {'question': 'What is the meaning of life?', 'answer': '42'},
        'planning_port': 1234
    }
    
    print("Running agent...")
    result = agent._run(data, model='mock-model')
    print("Agent finished.")
    
    # Check logs for FINAL_STATS
    log_files = os.listdir("logs")
    found_stats = False
    
    for log_file in log_files:
        if log_file.startswith("tokens_"):
            with open(os.path.join("logs", log_file), "r") as f:
                content = f.read()
                print(f"Log content:\n{content}")
                if "FINAL_STATS" in content:
                    found_stats = True
                    # Check for expected counts
                    # 1 search, 1 visit, 3 rounds (search, visit, answer) -> 3 input/output updates
                    # input tokens: 3 * 10 = 30
                    # output tokens: 3 * 5 = 15
                    if "search_calls=1" in content and "visit_calls=1" in content:
                        print("SUCCESS: Tool call counts are correct.")
                    else:
                        print("FAILURE: Tool call counts are incorrect.")
                        
                    if "orch_input_tokens=30" in content: # 3 calls * 10 tokens
                         print("SUCCESS: Input token count is correct.")
                    else:
                         print(f"FAILURE: Input token count is incorrect. Expected 30.")

                    if "orch_output_tokens=15" in content: # 3 calls * 5 tokens
                         print("SUCCESS: Output token count is correct.")
                    else:
                         print(f"FAILURE: Output token count is incorrect. Expected 15.")

    if found_stats:
        print("SUCCESS: FINAL_STATS found in logs.")
    else:
        print("FAILURE: FINAL_STATS not found in logs.")

if __name__ == "__main__":
    test_logging()
