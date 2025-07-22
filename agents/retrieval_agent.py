import json
from datetime import datetime
from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
# Import the dedicated re-ranking library
from flashrank import Ranker, RerankRequest

class RetrievalAgent:
    """
    Retrieves documents using a fetch-and-rerank strategy for higher accuracy.
    This agent searches across all available domains and uses a dedicated re-ranking
    model to find the most relevant documents before passing them to the LLM.
    """

    def __init__(self, llm, retrievers: Dict[str, VectorStoreRetriever], memory, top_k_rerank: int = 5):
        self.llm = llm
        self.retrievers = retrievers
        self.memory = memory
        self.reflection_message = ""
        self.log_history = []
        # The number of top documents to keep after re-ranking
        self.top_k_rerank = top_k_rerank
        # Initialize the re-ranker. ms-marco-MiniLM-L-12-v2 is a good, lightweight model.
        try:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt/flashrank-cache")
        except Exception as e:
            print(f"Warning: Could not initialize flashrank Ranker. Re-ranking will be skipped. Error: {e}")
            self.reranker = None


    def log(self, role, content):
        self.log_history.append({"role": role, "content": content})

    def _save_log(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/ra_session_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump({
                "log": self.log_history,
                "reflection": self.reflection_message
            }, f, indent=2)

    def update_reflection(self, success):
        if success:
            self.reflection_message += " Found relevant documents via re-ranking."
        else:
            self.reflection_message += " Could not find useful documents even after re-ranking."

    def process_request(self, query: str) -> AIMessage:
        self.log("ca_message", query)
        self.log("ra_thought", f"Step 1: Fetching initial documents for query: '{query}'")

        # Step 1: Fetch a larger set of initial documents from all retrievers.
        all_docs: List[Document] = []
        for domain, retriever in self.retrievers.items():
            try:
                # Configure the retriever to fetch more documents (e.g., 25) to give the re-ranker more to work with.
                retriever.search_kwargs = {'k': 25}
                retrieved_docs = retriever.invoke(query)
                if retrieved_docs:
                    self.log(f"retrieved_{len(retrieved_docs)}_docs_from_{domain}", [doc.page_content for doc in retrieved_docs])
                    all_docs.extend(retrieved_docs)
            except Exception as e:
                self.log("error", f"Error retrieving from domain {domain}: {e}")

        if not all_docs:
            self.update_reflection(False)
            self._save_log()
            return AIMessage(content="NO_INFO_FOUND: No documents were found in the initial fetch.")

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        self.log("initial_unique_docs_count", len(unique_docs))

        # Step 2: Re-rank the retrieved documents to find the most relevant ones.
        if self.reranker:
            rerank_request = RerankRequest(query=query, passages=[{"text": doc.page_content} for doc in unique_docs])
            reranked_results = self.reranker.rerank(rerank_request)
            top_docs_content = [result['text'] for result in reranked_results[:self.top_k_rerank]]
            self.log("reranked_top_docs_content", top_docs_content)
        else:
            # Fallback if re-ranker failed to initialize
            self.log("warning", "Re-ranker not available, using top documents from initial fetch.")
            top_docs_content = [doc.page_content for doc in unique_docs[:self.top_k_rerank]]


        if not top_docs_content:
            self.update_reflection(False)
            self._save_log()
            return AIMessage(content="NO_INFO_FOUND: Re-ranking did not find any relevant documents.")

        # Step 3: Use the LLM to synthesize an answer from ONLY the top-ranked documents.
        context = "\n\n---\n\n".join(top_docs_content)
        
        summary_prompt = f"""You are a helpful assistant. Based ONLY on the following highly relevant documents, provide a direct and concise answer to the user's original query.

        User's Original Query: "{query}"

        Retrieved Documents:
        ---
        {context}
        ---

        If the documents contain the answer, extract it precisely. If they do not, respond with exactly "NO_INFO_FOUND".
        """
        
        summary_response = self.llm.invoke(summary_prompt)
        summary = summary_response.content.strip()

        if "no_info_found" in summary.lower():
            self.update_reflection(False)
            self._save_log()
            return AIMessage(content="NO_INFO_FOUND")

        self.log("ra_summary", summary)
        self.update_reflection(True)
        self._save_log()
        return AIMessage(content=summary)
