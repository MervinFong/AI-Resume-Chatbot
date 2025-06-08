from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

from chatbot.resume_analyzer import analyze_resume
from chatbot.firebase_utils import fetch_resume_from_firebase, fetch_mbti_from_firebase
from job_scrapers.jobstreet_wrapper import fetch_scraped_jobs

# 🔧 Step 1: Tool Definitions

def get_resume(email: str) -> str:
    return fetch_resume_from_firebase(email)

def get_mbti(email: str) -> str:
    return fetch_mbti_from_firebase(email)

def analyze_user_resume(email: str) -> str:
    resume_text = fetch_resume_from_firebase(email)
    result = analyze_resume(resume_text)
    return f"Here is the resume analysis:\n\n{result}"

def get_job_recommendations(context: str) -> str:
    return fetch_scraped_jobs(context)

# 🔨 Step 2: Tool List for Agent

tools = [
    Tool(
        name="GetUserResume",
        func=get_resume,
        description="Fetch the user's uploaded resume text from the system using their email"
    ),
    Tool(
        name="AnalyzeResume",
        func=analyze_user_resume,
        description="Analyze the user's resume and give improvement suggestions"
    ),
    Tool(
        name="GetMBTI",
        func=get_mbti,
        description="Fetch the user's MBTI result based on their email"
    ),
    Tool(
        name="GetJobRecommendations",
        func=get_job_recommendations,
        description="Scrape jobs based on a provided context, such as resume text or MBTI type"
    ),
]

# 🤖 Step 3: Initialize GPT Agent

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 🧪 Step 4: Use Agent in Code

def run_smart_chat(email: str, user_input: str):
    prefix = f"The user's email is {email}. Answer based on their resume, MBTI, and job data if needed."
    return agent.run(f"{prefix} {user_input}")
