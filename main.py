from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from duckduckgo_search import DDGS

# Setup local model via Ollama
llm = Ollama(model="mistral")

# Web search tool using DuckDuckGo
def search_duckduckgo(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return "\n".join(r["body"] for r in results[:3])

search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_duckduckgo,
    description="Search the web for recent information."
)

# Create the agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run CLI loop
while True:
    q = input("\n❓ Ask the agent something (or type 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    answer = agent.run(q)
    print("\n🧠 Agent Response:\n", answer)
