import json
from langchain_core.messages import AIMessage, HumanMessage

class OptimizerAgent:
    """
    An agent that analyzes a failed conversation and the judge's feedback
    to suggest improvements to another agent's prompt.
    """
    def __init__(self, model):
        self.model = model

    def suggest_improvement(self, conversation_log, judge_score, judge_judgment, original_prompt):
        """
        Takes a failed conversation and suggests a new, improved prompt.
        """
        # Ensure the log is in a readable format
        serializable_log = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in conversation_log
        ]
        
        prompt = self._build_prompt(serializable_log, judge_score, judge_judgment, original_prompt)
        
        try:
            response_object = self.model.invoke(prompt)
            content = response_object.content.strip()
            # We expect the output to be the new prompt itself
            return content
        except Exception as e:
            return f"Error generating suggestion: {e}"

    def _build_prompt(self, conversation_log, judge_score, judge_judgment, original_prompt):
        """Builds a detailed prompt for the LLM to optimize another agent's prompt."""
        
        log_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_log])
        
        prompt = f"""You are an expert System Prompt Engineer. Your task is to improve the prompt of an AI agent that has failed.

        **Analysis of Failure**
        An AI agent, the 'CommunicationAgent', had the following interaction which was rated poorly by a 'JudgeAgent'.

        **Conversation Log:**
        ---
        {log_str}
        ---

        **Judge's Feedback:**
        - Score: {judge_score}
        - Judgment: {judge_judgment}

        **The Agent's Original Flawed Prompt:**
        ---
        {original_prompt}
        ---

        **Your Task:**
        Based on the conversation log and the judge's feedback, identify the core flaw in the original prompt's logic. Rewrite the prompt to be more robust and prevent this specific type of failure in the future. 
        
        Output ONLY the new, improved prompt. Do not include any explanation, preamble, or markdown formatting.
        """
        return prompt

