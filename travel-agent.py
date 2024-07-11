import os
import bs4

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence 

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

query = """
  Vou viajar para Londres em 2024. 
  Quero que faças um roteiro de viagem para mim com os eventos que irão ocorrer na data da viagem e com o preço do bilhete de avião do Porto para Londres com a companhia da easyjet (preço em euros €).
"""

def research_agent(query, llm):
  tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
  prompt = hub.pull("hwchase17/react")
  agent = create_react_agent(llm, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
  context = agent_executor.invoke({"input": query})
  return context['output']


def load_data():
  loader = WebBaseLoader(
    web_paths=("https://www.dicasdeviagem.com/inglaterra",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading backgroud-image loading-dark")))
  )
  docs = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=api_key))
  retriever = vectorstore.as_retriever()
  return retriever


def get_relevant_docs(query):
  retriever = load_data()
  relevant_docs = retriever.invoke(query)
  return relevant_docs

def supervisor_agent(query, llm, context, relevant_docs):
  prompt_template = """
    Tu és o responsável de uma agência de viagens. A tua resposta deve ser um roteiro de viagem completo e detalhado.
    Utiliza o contexto do evento ,preço de viagens, input do uilizador e também os documentos relevantes.
    Contexto: {context}
    Documentos relevantes: {relevant_docs}
    Utilizador: {query}
    Assistente: 
  """

  prompt = PromptTemplate(
    input_variables=['context', 'relevant_docs', 'query'],
    template = prompt_template
  )

  sequence = RunnableSequence(prompt | llm)

  response = sequence.invoke({"context": context, "relevant_docs": relevant_docs, "query": query})
  return response

def get_response(query, llm):
  context = research_agent(query, llm)
  relevant_docs = get_relevant_docs(query)
  response = supervisor_agent(query, llm, context, relevant_docs)
  return response

print(get_response(query, llm).content)