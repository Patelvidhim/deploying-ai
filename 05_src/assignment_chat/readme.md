# Astra — The Research Assistant

Astra is an AI-powered research assistant designed to help users explore, summarize, and analyze academic research efficiently. It combines multiple services to provide a conversational interface for literature review and semantic search.

---

## Features

### 1. ArXiv API Retrieval & Summarization
- Fetches academic papers from ArXiv based on user queries.
- Parses XML responses and generates **mini literature reviews** using GPT.
- Provides concise, human-readable summaries of relevant papers.

### 2. Local Semantic Search
- Searches a local research dataset (15,000 rows) using **ChromaDB embeddings**.
- Retrieves semantically relevant papers with metadata: title, authors, publication date, category.
- Summarizes results for easy understanding.

### 3. Web Search & Summarization
- Performs live **OpenAI Web Search** for the latest research trends.
- Summarizes web findings into coherent research insights.

---

## How It Works

User
│
│ │
│ ├─ ArXiv API → XML → LLM Summary
│ ├─ ChromaDB Semantic Search → Metadata + Summary
│ └─ OpenAI Web Search → LLM Summary
│
└─► Response (structured & human-readable)

 Maintains **session memory** for context-aware conversation.
- Provides guidance for effective research queries.
- Implements **guardrails**: blocks topics like Taylor Swift, cats, dogs, and horoscopes.

---

## Technical Stack

| Component         | Technology                         |
|------------------|-----------------------------------|
| Language Model    | GPT-4.1 & GPT-4.1-mini            |
| Interface         | Gradio                            |
| Embeddings        | OpenAI `text-embedding-3-small`   |
| Vector Database   | ChromaDB (persistent client)      |
| APIs              | ArXiv XML API, OpenAI Web Search  |
| Backend           | Python, Pandas, Requests, XML parsing |
