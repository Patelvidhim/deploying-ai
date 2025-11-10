# app.py
from assignment_chat.main import get_assignment_chat_agent
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv
import os

from utils.logger import get_logger

# Initialize logger
_logs = get_logger(__name__)

# Load environment variables
load_dotenv('.secrets')  # Make sure you run this from the top folder of the app

# Check OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Initialize LLM agent
llm = get_assignment_chat_agent()

# Chat function
def assignment_chat(message: str, history: list[dict]) -> str:
    langchain_messages = []
    n = 0
    _logs.debug(f"History: {history}")

    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
            n += 1

    # Append current user message
    langchain_messages.append(HumanMessage(content=message))

    state = {
        "messages": langchain_messages,
        "llm_calls": n
    }

    response = llm.invoke(state)
    return response['messages'][-1].content

# Initial message for the chatbot
initial_message = (
    "Hi, I am Jarvis your research assistant!\n\n"
    "I can help you with:\n"
    "- Searching papers directly from arXiv and giving literature reviews\n"
    "- Performing a semantic search using my local database (15000 papers) — mention my local db if you want searches performed there\n"
    "- Querying the web and summarising results for scientific and research areas\n\n"
    "You can ask me things like:\n"
    "- 'Show me the top papers on time series forecasting in 2025.'\n"
    "- 'Find papers on attention mechanisms in forecasting.'\n"
    "- 'Do a web search for the most recent trends on time series forecasting and summarise your results.'\n\n"
    "What would you like to do?"
)

# Create Gradio chat interface
chat = gr.ChatInterface(
    fn=assignment_chat,
    type="messages",
    chatbot=gr.Chatbot(placeholder=initial_message, type='messages'),
)

# Launch the app
if __name__ == "__main__":
    _logs.info("Starting Assignment Chat App...")
    chat.launch()
