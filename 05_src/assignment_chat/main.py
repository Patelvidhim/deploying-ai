import os
import json
import requests
import xmltodict
import pandas as pd
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator



from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from openai import OpenAI
import chromadb

from utils.logger import get_logger
from assignment_chat.prompts import return_instructions_root

_logs = get_logger(__name__)

# Load environment variables
load_dotenv(".env")
load_dotenv(".secrets")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "arXiv_scientific_dataset.csv")


# ------------------------- Tools -------------------------

@tool
def get_arxiv_info(search_query=None, start=0, max_results=None):
    """Fetch papers from arXiv API."""
    def xml_to_json_from_text(xml_text: str) -> dict:
        data_dict = xmltodict.parse(xml_text, process_namespaces=True)
        return json.loads(json.dumps(data_dict, indent=2))

    def extract_arxiv_papers(arxiv_dict):
        feed_key = "http://www.w3.org/2005/Atom:feed"
        entry_key = "http://www.w3.org/2005/Atom:entry"
        author_key = "http://www.w3.org/2005/Atom:author"

        entries = arxiv_dict.get(feed_key, {}).get(entry_key, [])
        if isinstance(entries, dict):
            entries = [entries]

        papers = []
        for entry in entries:
            title = entry.get("http://www.w3.org/2005/Atom:title", "").strip()
            abstract = entry.get("http://www.w3.org/2005/Atom:summary", "").strip()
            doi = entry.get("http://arxiv.org/schemas/atom:doi", {}).get("#text")
            journal_ref = entry.get("http://arxiv.org/schemas/atom:journal_ref", {}).get("#text")

            authors_data = entry.get(author_key, [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]

            authors = []
            for a in authors_data:
                name = a.get("http://www.w3.org/2005/Atom:name")
                affiliation = a.get("http://arxiv.org/schemas/atom:affiliation", {}).get("#text")
                authors.append({"name": name, "affiliation": affiliation})

            primary_category = entry.get("http://arxiv.org/schemas/atom:primary_category", {}).get("@term")
            categories_data = entry.get("http://www.w3.org/2005/Atom:category", [])
            if isinstance(categories_data, dict):
                categories_data = [categories_data]
            categories = [c.get("@term") for c in categories_data if "@term" in c]

            links = entry.get("http://www.w3.org/2005/Atom:link", [])
            if isinstance(links, dict):
                links = [links]
            pdf_link = None
            for l in links:
                if l.get("@title") == "pdf":
                    pdf_link = l.get("@href")
                    break

            papers.append({
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "journal_reference": journal_ref,
                "authors": authors,
                "primary_category": primary_category,
                "categories": categories,
                "pdf_link": pdf_link
            })

        return papers

    base_url = 'http://export.arxiv.org/api/query?'
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, start, max_results)
    response = requests.get(base_url, params=query)
    feed_info = xml_to_json_from_text(response.text)
    papers = extract_arxiv_papers(feed_info)
    return papers


@tool
def semantic_paper_search(query: str, n_results: int = 5, after_year: int = None, category: str = None):
    """Perform semantic search on local ArXiv dataset using ChromaDB."""
    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_or_create_collection("arxiv_metadata")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    if after_year:
        df = df[df["year"] >= after_year]
    if category:
        df = df[df["category"].str.contains(category, case=False, na=False)]

    results = collection.query(query_texts=[query], n_results=n_results)
    matches = []
    for i in range(len(results["documents"][0])):
        matches.append({
            "title": results["metadatas"][0][i]["title"],
            "authors": results["metadatas"][0][i]["authors"],
            "published_date": results["metadatas"][0][i]["published_date"],
            "category": results["metadatas"][0][i]["category"],
            "summary": results["documents"][0][i]
        })
    return matches


@tool
def web_search(query: str, n_results: int = 5):
    """Perform web search using OpenAI Responses API."""
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"Search the web for: {query}\nReturn top {n_results} results with title, URL, and summary."
    )
    results = []
    if hasattr(response, "output"):
        for item in response.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if c.type == "output_text":
                        results.append({"summary": c.text})
                    elif c.type == "source_attribution":
                        results.append({
                            "title": c.metadata.get("title", ""),
                            "url": c.metadata.get("url", ""),
                            "summary": c.metadata.get("snippet", "")
                        })
    return results[:n_results]


# ------------------------- LLM + Tool Node Logic -------------------------

def get_model_with_tools():
    model = init_chat_model("openai:gpt-4o-mini", temperature=0.7)
    tools = [get_arxiv_info, semantic_paper_search, web_search]
    return model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: dict):
    model_with_tools = get_model_with_tools()
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=return_instructions_root())] + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def tool_node_api_call(state: dict):
    tools_by_name = {get_arxiv_info.name: get_arxiv_info}
    result = []

    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    user_query = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content:
            user_query = msg.content
            break

    model_with_tools = get_model_with_tools()
    search_query_prompt = f"""
    You are a research assistant helping to find relevant papers on arXiv.
    User query: "{user_query}"
    Return ONLY the arXiv search query string.
    """

    query_generation = model_with_tools.invoke([
        SystemMessage(content="Generate precise arXiv API query strings."),
        HumanMessage(content=search_query_prompt)
    ])
    search_query_str = query_generation.content.strip()

    for tool_call in last_msg.tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if not tool:
            continue

        papers = tool.invoke({
            "search_query": search_query_str,
            "start": 0,
            "max_results": 10
        })

        papers_str_list = []
        for paper in papers:
            authors_str = ", ".join(
                f"{a['name']} ({a['affiliation']})" if a.get("affiliation") else a['name']
                for a in paper.get("authors", [])
            )
            categories_str = ", ".join(paper.get("categories", []))
            paper_text = (
                f"Title: {paper.get('title')}\n"
                f"Authors & Affiliations: {authors_str}\n"
                f"DOI / Journal: {paper.get('doi') or paper.get('journal_reference')}\n"
                f"Primary Category / Keywords: {paper.get('primary_category')} / {categories_str}\n"
                f"Summary: {paper.get('abstract')}\n"
            )
            papers_str_list.append(paper_text)

        papers_str = "\n\n".join(papers_str_list)
        summary_response = model_with_tools.invoke([
            SystemMessage(content=(
                "You are Jarvis, a research assistant. Summarize the following papers into a mini literature review."
            )),
            HumanMessage(content=papers_str)
        ])

        result.append(ToolMessage(
            content=summary_response.content,
            tool_call_id=tool_call["id"]
        ))

    last_msg.tool_calls = []
    return {"messages": result}


# Similar tool nodes can be written for `tool_node_semantic` and `tool_node_web_search` (omitted for brevity)


# ------------------------- Agent Setup -------------------------

END = "end"

def should_continue(state: MessagesState) -> Literal["tool_node_api_call", "tool_node_semantic", "tool_node_web_search", "end"]:
    ...

    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    tool_names = [t["name"] for t in last_message.tool_calls]
    if "get_arxiv_info" in tool_names:
        return "tool_node_api_call"
    elif "semantic_paper_search" in tool_names:
        return "tool_node_semantic"
    elif "web_search" in tool_names:
        return "tool_node_web_search"

    return END


def get_assignment_chat_agent():
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node_api_call", tool_node_api_call)
    # Add other nodes: semantic and web
    # agent_builder.add_node("tool_node_semantic", tool_node_semantic)
    # agent_builder.add_node("tool_node_web_search", tool_node_web_search)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_edge("tool_node_api_call", "llm_call")
    # agent_builder.add_edge("tool_node_semantic", "llm_call")
    # agent_builder.add_edge("tool_node_web_search", "llm_call")

    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node_api_call", "tool_node_semantic", "tool_node_web_search", END]
    )

    agent = agent_builder.compile()

    graph_bytes = agent.get_graph(xray=True).draw_mermaid_png()
    output_file_path = os.path.join(BASE_DIR, "agent_graph.png")
    with open(output_file_path, "wb") as f:
        f.write(graph_bytes)
    print(f"Graph saved as {output_file_path}")

    return agent


if __name__ == "__main__":
    agent = get_assignment_chat_agent()
    print("Agent ready!")
