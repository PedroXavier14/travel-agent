import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))

tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)

agent = initialize_agent(
  tools,
  llm,
  agent = 'zero-shot-react-description',
  verbose=True
)

print("------------------------")
#print(agent.agent.llm_chain.prompt.template)

query = """
  Vou viajar para Londres em 2024. 
  Quero que faças um roteiro de viagem para mim com os eventos que irão ocorrer na data da viagem e com o preço do bilhete de avião do Porto para Londres com a companhia da easyjet (preço em euros).
"""

agent.run(query)