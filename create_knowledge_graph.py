from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from util_module import get_text_from_files, add_all_docs_to_graph, createHybridIndex, createEntityIndex

load_dotenv() #load environment vars

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]

#knowledge graph object
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

llm_transformer = LLMGraphTransformer(llm=llm)
FILE_PATH = f"{os.getcwd()}/book"


#read all files in the path

text = get_text_from_files(FILE_PATH)
print('text reading completed successfully')

#add everything to graph

add_all_docs_to_graph(text, kg, llm_transformer)
print(' docs is added to graph succeessfully')

#create index on the text to perform keyword search and similarity search both
createHybridIndex()

#create index on each node

createEntityIndex(kg)

print("=======================END==========================")