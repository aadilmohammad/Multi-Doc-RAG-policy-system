Full RAG System – Query Rewrite + Multi‑Query Retrieval
This project implements a complete Retrieval‑Augmented Generation (RAG) pipeline with query rewriting, multi‑query document retrieval, FAISS vector search, and LLM‑based answer evaluation. It is designed as a production‑style RAG system that can ingest documents, chunk them, embed them, build a FAISS index, retrieve relevant context, generate answers, and evaluate those answers using an LLM judge.

The system is modular, extensible, and structured like a real enterprise RAG backend. It includes ingestion, chunking, embedding, indexing, retrieval, query rewriting, answer generation, and evaluation components.

Features
• Document ingestion from a folder
• Text chunking with configurable size and overlap
• Embedding generation using a transformer model
• FAISS vector index creation and storage
• Multi‑query retrieval using rewritten queries
• Query rewriting using an LLM
• Answer generation using retrieved context
• LLM‑based evaluation (faithfulness, groundedness, relevance, completeness)
• CLI chatbot interface
• Clean modular architecture suitable for production or deployment

Project Structure
rag/
Contains all core modules: ingestion, chunking, embeddings, index building, retriever, query rewriting, pipeline, and evaluation.

scripts/
Contains runnable scripts for building the index and running the chatbot.

docs/
Contains the text documents used for retrieval.

data/
Stores the FAISS index and metadata generated during ingestion.

How It Works
Documents are loaded from the docs folder.

Each document is chunked into overlapping text segments.

Chunks are embedded using a transformer embedding model.

FAISS index is built from the embeddings.

During chat, the user query is rewritten into multiple alternative queries.

Each rewritten query is embedded and searched in FAISS.

Retrieved chunks are merged, deduplicated, and ranked.

The final answer is generated using the retrieved context.

The answer is evaluated using an LLM judge.

Running the System
Step 1: Build the FAISS index
python -m scripts.build_index

Step 2: Start the chatbot
python -m scripts.chat_CLI

Example Queries
What is the laptop replacement policy for remote employees
How do I reset my VPN password
What are the MFA requirements
How do I escalate an unresolved IT ticket
Explain all security requirements for remote workers

Documents Included
HR Remote Work Policy
IT Security and Access Policy
VPN and Authentication Guidelines
Employee Helpdesk Procedures

These documents simulate a realistic enterprise knowledge base for testing retrieval quality.

Environment Variables
The system requires a Groq API key for query rewriting, answer generation, and evaluation.
Set your key in a .env file:

GROQ_API_KEY=your_key_here

The project automatically loads the .env file.

Requirements
Python 3.10+
FAISS
Transformers / Sentence Embeddings
Groq API client
python-dotenv

Install dependencies using:

pip install -r requirements.txt

Evaluation
The system evaluates each answer using an LLM judge based on:

• Faithfulness
• Groundedness
• Relevance
• Completeness

The evaluation is returned as a JSON object.

