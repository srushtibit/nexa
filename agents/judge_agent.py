import re
from langchain_core.messages import AIMessage, HumanMessage

class JudgeAgent:
    """
    An intelligent agent that evaluates the quality of a conversation log
    based on a set of predefined criteria.
    """
    def __init__(self, model):
        self.model = model

    def evaluate(self, conversation_log):
        """
        Evaluates a conversation log using the provided LLM.
        """
        # Ensure the log is in a format the LLM can process
        serializable_log = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in conversation_log
        ]
        
        prompt = self._build_prompt(serializable_log)
        
        try:
            # The ChatGroq model returns an AIMessage object
            response_object = self.model.invoke(prompt)
            content = response_object.content.strip()

            # Use regex to robustly parse the score and judgment
            score_match = re.search(r"Score:\s*([0-9.]+)", content)
            judgment_match = re.search(r"Judgment:\s*(.*)", content, re.DOTALL)

            if not score_match or not judgment_match:
                # If parsing fails, return a specific error
                return 0.0, f"Evaluation failed: Could not parse score/judgment from model output: '{content}'"

            score_str = score_match.group(1)
            judgment = judgment_match.group(1).strip()

            score = float(score_str)

            return score, judgment

        except Exception as e:
            return 0.0, f"Error during evaluation: {e}"

    def _build_prompt(self, conversation_log):
        """Builds a detailed prompt for the LLM to evaluate the conversation."""
        
        # Convert the list of dicts into a readable string
        log_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_log])
        
        prompt = f"""You are an impartial judge evaluating an AI agent interaction for employees at **NexaCorp**. The agent's primary goal is to solve internal company issues using its tools.

        **Evaluation Criteria:**
        1.  **Correctness & Faithfulness:** Was the final answer factually correct based *only* on the information from the tools? (Score 1-3 for wrong answers, 4-7 for partially correct, 8-10 for perfect).
        2.  **Tool Usage:** Did the agent choose the right tool for the job? Did it avoid using tools for simple conversation? (Deduct points for incorrect tool use).
        3.  **Efficiency:** Did the agent solve the problem without unnecessary steps or loops? (Deduct points for inefficiency).
        4.  **Clarity:** Was the final answer clear, direct, and helpful to the user?

        **Conversation Log to Evaluate:**
        ---
        {log_str}
        ---

        Based on the criteria above, provide your evaluation in the following format ONLY. Do not add any other text or explanation.
        Score: <a single score from 1.0 to 10.0>
        Judgment: <a brief, one-sentence judgment summarizing the agent's performance>
        """
        return prompt
