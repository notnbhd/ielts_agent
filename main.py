from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

llm = ChatOllama(model="llama3.2", temperature=0)
tools = [DuckDuckGoSearchRun()]

agent = create_react_agent(llm, tools)

result = agent.invoke({"messages": [("user", "LangGraph là gì?")]})
print(result["messages"][-1].content)
