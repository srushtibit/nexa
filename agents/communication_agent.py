import json
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
import re

class CommunicationAgent:
    """
    A stateful coordinator agent that maintains conversation history to handle
    multi-turn dialogues and make more intelligent decisions.
    """
    def __init__(self, llm, tools: dict, memory, max_turns=5):
        self.llm = llm
        self.tools = tools
        self.memory = memory # Though unused, kept for structural consistency
        self.reflection_message = ""
        # The log_history will now serve as our primary conversation memory
        self.log_history = []
        self.max_turns = max_turns

    def log(self, role, content):
        """Logs a message to the conversation history."""
        # We now store LangChain message objects for better structure
        if role == "user":
            self.log_history.append(HumanMessage(content=content))
        else:
            # For system/agent messages, we can use AIMessage or a custom format
            self.log_history.append(AIMessage(content=content))

    def _save_log_to_file(self, original_query):
        """Saves the structured log to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/ca_session_{timestamp}.json"
        
        # Convert message objects to a serializable format
        serializable_log = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in self.log_history
        ]

        with open(filename, "w") as f:
            json.dump({
                "initial_user_query": original_query,
                "log": serializable_log,
                "final_reflection": self.reflection_message
            }, f, indent=2)

    def handle_user_query(self, query):
        # Add the user's new message to the conversation history
        self.log("user", query)
        
        # The agent loop now runs for a few turns to decide on an action
        for i in range(self.max_turns):
            # The prompt now receives the entire conversation history
            prompt = self._build_prompt(self.log_history)
            
            model_response_object = self.llm.invoke(prompt)
            model_output = model_response_object.content.strip()
            
            self.log("assistant", f"[Thought]: {model_output}") # Log the thought process

            if "ANSWER:" in model_output:
                final_answer = model_output.split("ANSWER:", 1)[1].strip()
                self.log("assistant", f"[Final Answer]: {final_answer}")
                self._save_log_to_file(self.log_history[0].content)
                # Return only the answer and the full log
                return final_answer, self.log_history

            try:
                tool_match = re.search(r"TOOL:\s*(\w+)\s*QUERY:\s*(.*)", model_output, re.DOTALL)
                if not tool_match:
                    raise ValueError("Output does not match ANSWER: or TOOL: ... QUERY: ... format")

                tool_name = tool_match.group(1).strip()
                tool_query = tool_match.group(2).strip()

                if tool_name in self.tools:
                    tool_to_use = self.tools[tool_name]
                    tool_response = tool_to_use.process_request(tool_query)
                    
                    if isinstance(tool_response, AIMessage):
                        response_str = tool_response.content
                    elif isinstance(tool_response, dict):
                         response_str = json.dumps(tool_response)
                    else:
                        response_str = str(tool_response)

                    # Add the tool result to the conversation history for the next turn
                    self.log("assistant", f"[Tool Result for {tool_name}]: {response_str}")
                else:
                    self.log("assistant", f"[Error]: Attempted to use unknown tool: {tool_name}")

            except Exception as e:
                self.log("assistant", f"[Error]: Could not parse or execute tool. Error: {e}")
        
        final_answer = "I'm sorry, but I seem to be stuck. Could you please try rephrasing your request?"
        self.log("assistant", f"[Final Answer]: {final_answer}")
        self._save_log_to_file(self.log_history[0].content)
        return final_answer, self.log_history

    def _build_prompt(self, conversation_history):
        available_tools = "\n".join([f"- `{name}`: {(tool.__class__.__doc__ or 'No description available').strip()}" for name, tool in self.tools.items()])

        # Convert conversation history to a readable string format
        history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in conversation_history])

        prompt = f"""You are a helpful assistant for **NexaCorp**. Your goal is to answer employee queries by using the correct tool based on the **entire conversation history**.

        **Your Thought Process:**
        1.  **Analyze the full Conversation History:** Understand the user's intent and any information they've provided in previous turns.
        2.  **Select a Tool OR Answer Directly:**
            - For general knowledge questions (HR, IT, Payroll policies), or to find similar past tickets, use the `retrieval_agent`.
            - For looking up a specific ticket by its ID, use the `ticket_tool`.
            - If the user is making small talk or you have enough information to answer, answer directly.
        3.  **Handle Failures:** If a tool fails, ask the user for clarification.

        **Available Tools:**
        {available_tools}

        **Conversation History:**
        ---
        {history_str}
        ---

        **Your Next Step:**
        Based on the full conversation, what is your next action?
        - If you have enough information to answer, respond with `ANSWER: <your final answer to the user>`.
        - If you need to use a tool, respond with `TOOL: <tool_name> QUERY: <query for the tool>`.

        Your decision:"""
        return prompt

