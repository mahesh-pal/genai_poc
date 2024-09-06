from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from names_entity_module import NamesEntitiy
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage

INDEX_NAME = 'harry_vector_index'
ENTITY_INDEX_NAME = 'harry_entity_index'

def createHybridIndex():
  vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(), 
    search_type="hybrid", # hybrid search means the search will be vector based as well as keyword based
    node_label="Document", # Node label to create index on
    index_name=INDEX_NAME, #index name
    text_node_properties=["text"],  # which property of node is used to create index on
    embedding_node_property="embedding",# embedding will be stored in this propert
)
  return vector_index

def get_hybrid_index():
  return Neo4jVector.from_existing_index(embedding=OpenAIEmbeddings(), index_name='vector', search_type="hybrid", keyword_index_name='keyword' )
  

def createEntityIndex(kg):
  kg.query(
   f"CREATE FULLTEXT INDEX {ENTITY_INDEX_NAME} IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
  

def get_text_from_files(path):
    files = os.listdir(path)
    docs = []
    for file in files:
        if not file.endswith('.txt'):
            continue
        filePath = f"{path}/{file}"
        text = TextLoader(filePath).load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        )
        d = text_splitter.split_documents(text)
        docs.append(d)
    return [item for sublist in docs for item in sublist]

def add_all_docs_to_graph(docs, kg, transformer):
    graph_documents = transformer.convert_to_graph_documents(docs)
    return kg.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)

    
def get_chain_to_extract_entity(llm):
    # this will extract  entity from the qiven question start
    entityPrompt = ChatPromptTemplate.from_messages(
        [
            (
             "system",
                "You are extracting person, character,concept, creature, Ghost, school house, Train, Vehicle organization entities from the text.",
            ),
            (
                "human",
               "do not return any description text.Please return only the entities as a list of names like ['entity1', 'entity2'] for the input: {question}"

            ),
        ]
        )

    return  entityPrompt | llm


#each entity is broken in smallar chunk 
def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines 
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def parseEntity(input_list):
    print("Original input:", input_list)
    
    # Remove the square brackets and any extra quotes
    content = input_list.strip('[]')
    print("Content without brackets:", content)
    
    # Split the content by commas and strip extra spaces and quotes from each item
    items = [item.strip().strip("'\"") for item in content.split(',')]
    
    print("Parsed items:", items)
    return NamesEntitiy(names=items)

def retrive_all_node_for_entity_in_question(question, entity_chain, kg) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    print(' inside retrive_all_node_for_entity_in_question')
    result = ""
    cont = entity_chain.invoke({"question": question}).content;
    print(cont)
    entities = parseEntity(cont)
    print(entities.names)
    for entity in entities.names:
        print(entity)
        print(generate_full_text_query(entity))

        response = kg.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
YIELD node, score
CALL {
    WITH node
    MATCH (node)-[r:!MENTIONS]->(neighbor)
    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
    UNION
    MATCH (node)<-[r:!MENTIONS]-(neighbor)
    RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
}
RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
        print(' ===printing structured entity=========')
        print(result)
    return result

def perform_search_on_kw_and_vector(question,entity_chain, kg):
    print(f"Search query: {question}")
    def inner(question):
      vector_index = get_hybrid_index()
      structured_data = retrive_all_node_for_entity_in_question(question, entity_chain, kg)
      unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
      final_data = f"""Structured data:
         {structured_data}
       Unstructured data:
      {"#Document ". join(unstructured_data)}
    """
      return final_data
    return inner


def format_chat_history(chat_history):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer