import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime
# === 11/5 4pm: START token logging additions ===
import hashlib  # for stable log file names
# === 11/5 4pm: END token logging additions ===
# === 11/5 4pm: START summary additions ===
import re
# === 11/5 4pm: END summary additions ===


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]

        # === 11/5 4pm: START token logging additions ===
        # Cache a tokenizer to avoid re-loading every time (perf) and a log path holder
        self._tokenizer = None
        self._token_log_path: Optional[str] = None
        # === 11/5 4pm: END token logging additions ===

        # === 11/5 4pm: START summary additions ===
        # Soft cap for context token pressure (summarize when exceeded)
        # Defaults: summarize a bit before your hard cap (110k).
        self._soft_token_cap = int(os.getenv("SOFT_TOKEN_CAP", "80000"))
        # How many recent messages (after [0]=system, [1]=question) to keep as tail when pruning
        self._summary_tail_keep = int(os.getenv("SUMMARY_TAIL_KEEP", "4"))
        # Max messages to feed into summarizer body (to keep the side-call cheap)
        self._summary_clip_body = int(os.getenv("SUMMARY_CLIP_BODY", "12"))
        # Minimal guard to avoid repeated summarize within same round if nothing changed
        self._last_summary_text: Optional[str] = None
        # === 11/5 4pm: END summary additions ===

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10):
        # Prefer agent-specific env; fall back to generic if needed
        openai_api_key  = os.getenv("DR_API_KEY")  or os.getenv("API_KEY")
        openai_api_base = os.getenv("DR_API_BASE") or os.getenv("API_BASE") or "https://openrouter.ai/api/v1"
        resolved_model  = os.getenv("DR_MODEL_NAME") or os.getenv("MODEL_NAME") or "alibaba/tongyi-deepresearch-30b-a3b"

        # If self.model is "api" or empty, use the real model id from env
        use_model = resolved_model if str(getattr(self, "model", "")).strip().lower() in ("", "api") else self.model

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=use_model,  # <--- use the resolved model
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = (chat_response.choices[0].message.content or "").strip()

                # Optional: prepend reasoning if provider returns it
                try:
                    reasoning_text = getattr(chat_response.choices[0].message, "reasoning", None)
                    if isinstance(reasoning_text, str) and reasoning_text.strip():
                        content = "<think>\n" + reasoning_text.strip() + "\n</think>" + content
                except Exception:
                    pass

                if content:
                    print("--- Service call successful, received a valid response ---")
                    return content
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        return "vllm server error!!!"



    def count_tokens(self, messages):
        # >>> minimal guard for API mode (no local HF tokenizer) <<<
        if str(getattr(self, "llm_local_path", "")).strip().lower() == "api":
            try:
                text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
            except Exception:
                text = str(messages)
            # cheap approx: ~1 token per 4 chars
            return max(1, len(text) // 4)

        # === 11/5 4pm: START token logging additions ===
        # Cache the tokenizer to avoid repeated loads (perf)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
        tokenizer = self._tokenizer
        # === 11/5 4pm: END token logging additions ===

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        
        return token_count

    # === 11/5 4pm: START token logging additions ===
    def count_text_tokens(self, text: str) -> int:
        """
        Count tokens for a plain text string. If running in API mode, approximate.
        """
        if str(getattr(self, "llm_local_path", "")).strip().lower() == "api":
            return max(1, len((text or "")) // 4)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
        toks = self._tokenizer(text or "", return_tensors="pt")
        return int(toks["input_ids"].shape[-1])

    def _init_token_logger(self, question: str):
        """
        Create a per-question log file under ./logs with a stable, readable name.
        """
        try:
            os.makedirs("logs", exist_ok=True)
            # name: tokens_{YYYYmmdd-HH%M%S}_{hash8}.log
            ts = time.strftime("%Y%m%d-%H%M%S")
            qhash = hashlib.sha256((question or "").encode("utf-8")).hexdigest()[:8]
            self._token_log_path = os.path.join("logs", f"tokens_{ts}_{qhash}.log")
            with open(self._token_log_path, "a", encoding="utf-8") as f:
                f.write(f"# Token log\n# question_hash={qhash}\n# created_at={ts}\n")
                f.write(f"# question_raw={question.strip().replace(os.linesep, ' ')[:2000]}\n\n")
        except Exception as _:
            self._token_log_path = None  # disable on failure

    def _write_token_log(self, line: str):
        """
        Append one line to the token log file (if initialized).
        """
        if not self._token_log_path:
            return
        try:
            with open(self._token_log_path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            pass
    # === 11/5 4pm: END token logging additions ===

    # === 11/5 4pm: START summary additions ===
    def _build_summary_prompt(self, clipped_messages: List[Dict]) -> List[Dict]:
        """
        Build a side-call prompt to compress context.
        We *do not* include messages[0] (system) and messages[1] (question) here.
        """
        system = (
            "You are a context compressor. Summarize the following conversation and tool outputs "
            "for continued problem solving. Be concise and faithful. No hidden reasoning. "
            "Do NOT include <tool_call>, <tool_response>, <answer>, or <think> tags."
        )
        instruction = (
            "Summarize the key task, findings, evidence, decisions, and open questions in ≤ 2000 characters. "
            "Use short bullets; each bullet ≤ 1 line. If a section is empty, write 'None'."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
            {"role": "user", "content": json.dumps(clipped_messages, ensure_ascii=False)}
        ]

    def summarize_messages(self, messages: List[Dict]) -> str:
        """
        Create a compact summary of the conversation so far (excluding [0]=system, [1]=question).
        Clips to last N body messages to control cost.
        """
        try:
            body = messages[2:] if len(messages) > 2 else []
            if not body:
                return ""
            if len(body) > self._summary_clip_body:
                body = body[-self._summary_clip_body:]
            mini_msgs = self._build_summary_prompt(body)
            text = self.call_server(mini_msgs, planning_port=None, max_tries=3)
            text = (text or "").strip()
            # sanitize accidental tags if any
            for tag in ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                        "<answer>", "</answer>", "<think>", "</think>"):
                text = text.replace(tag, "")
            # very small safeguard to avoid repeating identical summaries
            if text and text != self._last_summary_text:
                self._last_summary_text = text
            return text
        except Exception as e:
            # keep the loop resilient
            return ""
    # === 11/5 4pm: END summary additions ===


    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

        # === 11/5 4pm: START token logging additions ===
        # Start a per-question token log
        self._init_token_logger(question)
        # Record initial context tokens
        try:
            init_tokens = self.count_tokens(messages)
            self._write_token_log(f"INIT | context_tokens={init_tokens}")
        except Exception:
            pass
        # === 11/5 4pm: END token logging additions ===

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                # === 11/5 4pm: START token logging additions ===
                try:
                    final_ctx = self.count_tokens(messages)
                    self._write_token_log(f"FINAL | context_tokens={final_ctx}")
                except Exception:
                    pass
                # === 11/5 4pm: END token logging additions ===
                return result
            round += 1
            num_llm_calls_available -= 1

            # === 11/5 4pm: START token logging additions ===
            # Measure input token count for this LLM call
            try:
                input_tok = self.count_tokens(messages)
            except Exception:
                input_tok = -1
            # === 11/5 4pm: END token logging additions ===

            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            # === 11/5 4pm: START token logging additions ===
            # Measure output token count for the assistant content we just received
            try:
                output_tok = self.count_text_tokens(content or "")
                self._write_token_log(f"ROUND {round} | input_tokens={input_tok} | output_tokens={output_tok}")
            except Exception:
                pass
            # === 11/5 4pm: END token logging additions ===

            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."

                    else:
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            # === 11/5 4pm: START summary additions ===
            # Summarize under context pressure (soft cap) *before* hitting hard cap
            if token_count > self._soft_token_cap:
                try:
                    before_tok = token_count
                    summary_text = self.summarize_messages(messages)
                    if summary_text:
                        summary_msg = {
                            "role": "system",
                            "content": "<orchestrator_summary>\n" + summary_text + "\n</orchestrator_summary>"
                        }
                        # Keep [0]=system, [1]=question pinned; keep a small tail for continuity
                        head = messages[:2] if len(messages) >= 2 else messages[:]
                        body = messages[2:] if len(messages) > 2 else []
                        tail = body[-self._summary_tail_keep:] if len(body) > self._summary_tail_keep else body
                        messages = head + [summary_msg] + tail
                        # Log effect in token log
                        try:
                            after_tok = self.count_tokens(messages)
                            self._write_token_log(
                                f"SUMMARIZE | before_tokens={before_tok} | after_tokens={after_tok} | summary_chars={len(summary_text)}"
                            )
                        except Exception:
                            pass
                except Exception:
                    # if summarization fails, proceed to hard-cap handling below
                    pass
            # === 11/5 4pm: END summary additions ===

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                # === 11/5 4pm: START token logging additions ===
                try:
                    final_ctx = self.count_tokens(messages)
                    self._write_token_log(f"FINAL | context_tokens={final_ctx}")
                except Exception:
                    pass
                # === 11/5 4pm: END token logging additions ===
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }

        # === 11/5 4pm: START token logging additions ===
        try:
            final_ctx = self.count_tokens(messages)
            self._write_token_log(f"FINAL | context_tokens={final_ctx}")
        except Exception:
            pass
        # === 11/5 4pm: END token logging additions ===

        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="../eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
