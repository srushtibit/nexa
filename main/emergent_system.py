import sys
import os
import uuid
from dotenv import load_dotenv

# Add project root to the Python path to find agent/tool modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.communication_agent import CommunicationAgent
from agents.retrieval_agent import RetrievalAgent
# We will create the JudgeAgent in the next step, for now we can comment it out or create a placeholder
# from agents.judge_agent import JudgeAgent 
from tools.ticket_tool import TicketLookupTool

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- MOCK OBJECTS FOR UTILS (Placeholders) ---
class ContextMemory:
    """A mock memory class."""
    pass

def log_interaction(session_id, user_query, response, log, score, judgment):
    """A mock logging function."""
    print("--- Interaction Logged ---")
    
class JudgeAgent:
    """A mock judge agent that returns a default score."""
    def __init__(self, model):
        pass
    def evaluate(self, log):
        return 0.0, "Judge not implemented."
# --- END MOCK OBJECTS ---

def main():
    """Main function to initialize and run the emergent agent system."""
    os.makedirs("logs", exist_ok=True)
    load_dotenv("config/settings.env")

    # --- System Configuration ---
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("FATAL ERROR: GROQ_API_KEY not found. Please set it in config/settings.env")
        sys.exit(1)

    # --- Initialize Models and Tools ---
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

    # --- Load Vector Stores ---
    try:
        domain_retrievers = {
            "hr": Chroma(persist_directory="data/chroma/hr", embedding_function=embedding_model).as_retriever(),
            "it": Chroma(persist_directory="data/chroma/it", embedding_function=embedding_model).as_retriever(),
            "payroll": Chroma(persist_directory="data/chroma/payroll", embedding_function=embedding_model).as_retriever(),
            "tickets": Chroma(persist_directory="data/chroma/tickets", embedding_function=embedding_model).as_retriever(),
        }
    except Exception as e:
        print(f"FATAL ERROR: Could not load Chroma databases: {e}")
        print("Please run 'build_vector_stores.py' first.")
        sys.exit(1)

    # --- Initialize Agents and Tools ---
    ca_memory = ContextMemory()
    ra_memory = ContextMemory()

    retrieval_agent = RetrievalAgent(llm=llm, retrievers=domain_retrievers, memory=ra_memory, top_k_rerank=5)
    ticket_tool = TicketLookupTool(csv_path="data/nexacorp_tickets.csv")

    tools = {
        "retrieval_agent": retrieval_agent,
        "ticket_tool": ticket_tool
    }

    communication_agent = CommunicationAgent(llm=llm, tools=tools, memory=ca_memory)
    judge_agent = JudgeAgent(model=llm)

    # --- Start Interactive Loop ---
    print("NexaCorp Assistant is ready. Type 'exit' or 'quit' to end the session.")
    while True:
        user_query = input("\n[You]: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        if user_query:
            run_interaction(user_query, communication_agent, judge_agent)

def run_interaction(user_query: str, ca: CommunicationAgent, judge: JudgeAgent):
    """Handles a single turn of the conversation."""
    session_id = str(uuid.uuid4())[:8]
    print(f"\nThinking...")
    
    response, log = ca.handle_user_query(user_query)
    score, judgment = judge.evaluate(log)
    
    print(f"\n[System]: {response}\n[Judge Score]: {score} - {judgment}")
    log_interaction(session_id, user_query, response, log, score, judgment)

if __name__ == "__main__":
    main()
