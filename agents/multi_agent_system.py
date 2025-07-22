from langchain_core.messages import HumanMessage, AIMessage
from communication_agent import CommunicationAgent
from retrieval_agent import RetrievalAgent
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from datetime import datetime
import json
import os

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Load Ollama LLM
llm = OllamaLLM(model="llama3")

# Load vector stores
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstores = {
    "hr": Chroma(persist_directory="chroma/hr", embedding_function=embedding).as_retriever(),
    "it": Chroma(persist_directory="chroma/it", embedding_function=embedding).as_retriever(),
    "payroll": Chroma(persist_directory="chroma/payroll", embedding_function=embedding).as_retriever(),
    "tickets": Chroma(persist_directory="chroma/tickets", embedding_function=embedding).as_retriever(),
}

# Initialize agents
ra = RetrievalAgent(llm=llm, retrievers=vectorstores, memory=None, top_k_rerank=5)
ca = CommunicationAgent(llm=llm, tools={"retrieval_agent": ra}, memory=None)

def run_multi_agent_conversation(user_query: str, max_turns=3):
    print(f"\n=== User Query ===\n{user_query}\n")

    # Step 1: CA handles the initial query
    answer, ra_requests = ca.handle_user_query(user_query)

    for turn in range(max_turns):
        if not ra_requests:
            break

        # Step 2: RA processes each CA internal message
        ra_responses = []
        for msg in ra_requests:
            response = ra.process_request(msg)
            ra_responses.append(response)

        # Step 3: CA handles the responses
        answer, ra_requests = ca.handle_ra_response(ra_responses)

        if answer:
            break

    print(f"\n=== Final Answer ===\n{answer}\n")
    save_interaction(user_query, ca, ra, answer)

def save_interaction(user_query, ca, ra, final_answer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/emergent_agent_interaction_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "user_query": user_query,
            "ca_logs": ca.log_history,
            "ca_reflection": ca.reflection_message,
            "ra_logs": ra.log_history,
            "ra_reflection": ra.reflection_message,
            "final_answer": final_answer,
        }, f, indent=2)
    print(f"[SUCCESS] Interaction saved to {filename}")


if __name__ == "__main__":
    # You can change the test queries here
    user_queries = [
        "How do I apply for sick leave?",
        "What is the process to request a new laptop?",
        "Where can I check my last 3 payslips?",
        "My VPN is not connecting. What should I do?"
    ]

    for query in user_queries:
        run_multi_agent_conversation(query)
