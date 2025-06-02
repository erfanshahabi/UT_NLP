
import nest_asyncio
nest_asyncio.apply()

import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import re
import pandas as pd
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import sqlite3
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
import Levenshtein
import atexit
from pydantic import BaseModel, Field
from typing import List
from langchain_core.tools import Tool
import getpass


os.environ["LLAMA_CLOUD_API_KEY"] = "llx-Aqvil4BxJqZsClIYPHEOHkPI85mrUnnXiPolZSH6pXpieJ7m"


parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="text")

file_extractor = {".pdf": parser}
file_path = ["/Users/erfanshahabi/Desktop/uni/NLP/NLP-CA6/Codes/The_New_Complete_Book_of_Foos.pdf"]

data_for_parse = SimpleDirectoryReader(input_files=file_path, file_extractor=file_extractor)
data_for_parse

documents =data_for_parse.load_data()

documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    length_function=len,
    is_separator_regex=False,
)

documents_list = []
page_number = 0
last_doc = None
for doc in documents:
    if last_doc is None or last_doc != doc.metadata["file_name"]:
        page_number = 1
        last_doc = doc.metadata["file_name"]
    else:
        page_number += 1

    texts = text_splitter.split_text(doc.text)
    for text in texts:
        item = {}
      
        item["text"] = text
        item["metadata_file_name"] = doc.metadata["file_name"]
        item["metadata_creation_date"] = doc.metadata["creation_date"]
        item["metadata_pagenumber"] = page_number
        documents_list.append(item)



df = pd.DataFrame(documents_list)
df

db = lancedb.connect(".lancedb")

embedding_model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="mps")

class ChunksOfData(LanceModel):
    text: str = embedding_model.SourceField()
    metadata_file_name: str
    metadata_creation_date: str
    metadata_pagenumber: int
    vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()

def df_to_dict_batches(df: pd.DataFrame, batch_size: int = 128):
    """
    Yields data from a DataFrame in batches of dictionaries.
    Each batch is a list of dict, suitable for LanceDB ingestion.
    """
    for start_idx in range(0, len(df), batch_size):
        end_idx = start_idx + batch_size
        # Convert the batch of rows to a list of dict
        batch_dicts = df.iloc[start_idx:end_idx].to_dict(orient="records")
        yield batch_dicts

tbl = db.create_table(
    "embedded_chunk77",
    data=df_to_dict_batches(df, batch_size=10),
    schema=ChunksOfData,
)


db_path = "/Users/erfanshahabi/Desktop/uni/NLP/NLP-CA6/Codes/food_orders.db"


conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("tables:", tables)

conn.close()


conn = sqlite3.connect(db_path)


query = "SELECT * FROM food_orders"
df_orders = pd.read_sql_query(query, conn)


print("`food_orders`:")
print(df_orders)

query = "SELECT * FROM foods"
df_foods = pd.read_sql_query(query, conn)

print("`foods`:")
print(df_foods)

conn.close()
import os
from langchain_openai import ChatOpenAI

AVALAI_BASE_URL = "https://api.avalai.ir/v1"
GPT_MODEL_NAME = "gpt-4o-mini"

gpt4o_chat = ChatOpenAI(model=GPT_MODEL_NAME,
                        base_url=AVALAI_BASE_URL,
                        api_key="aa-NYbGWi06TiUnxZCpKCHTNyzrsSlLAerJlVfRknrCuNcGkic2")



class FoodRecommendationOutput(BaseModel):
    """
    Structured output for food recommendations.
    """
    food_type: str = Field(description="The inferred general category of food from the user's request.")
    specific_dishes: list[str] = Field(description="A list of specific dish names within this category.")
    reasoning: str = Field(description="Explanation of why this food type was chosen.")

def food_recommendation(user_query: str):
    """
    Infers food preferences using LLM, then searches the database for multiple matching dishes.

    :param user_query: The user's request for food suggestions.
    :return: A natural language response with recommended dishes, restaurant names, and prices.
    """

    print("LLM is analyzing:", user_query)

    analysis_prompt = (
        "The user wants food recommendations. Determine the general type of food they are looking for, "
        "and provide a list of 3-5 specific dish names that belong to this category.\n"
        "User request: {user_query}\n"
        "Respond in structured JSON format."
    )

    llm_with_structured_output = gpt4o_chat.with_structured_output(FoodRecommendationOutput)
    response = llm_with_structured_output.invoke(user_query)

    food_type = response.food_type
    specific_dishes = response.specific_dishes
    reasoning = response.reasoning
    print(f"Inferred food type: {food_type}")
    print(f" Reasoning: {reasoning}")
    print(f" Searching for: {specific_dishes}")


    connection = sqlite3.connect('/Users/erfanshahabi/Desktop/uni/NLP/NLP-CA6/Codes/food_orders.db')
    cursor = connection.cursor()

    query = """
        SELECT food_name, restaurant_name, price
        FROM foods
        WHERE food_category LIKE ?
        ORDER BY RANDOM()
        LIMIT 5;
    """

    cursor.execute(query, (f"%{food_type}%",))
    results = cursor.fetchall()

    if not results:
        placeholders = ', '.join('?' * len(specific_dishes))
        query = f"""
            SELECT food_name, restaurant_name, price
            FROM foods
            WHERE food_name IN ({placeholders})
            ORDER BY RANDOM()
            LIMIT 5;
        """
        cursor.execute(query, specific_dishes)
        results = cursor.fetchall()

    connection.close()


    if not results:
        return (
            f"Based on your request, I suggest trying **{food_type}** dishes. "
            "Unfortunately, I couldn't find exact matches in my database, but you can check nearby restaurants!"
        )

    response_text = f" Based on your preference, here are some **{food_type}** dishes available:\n\n"
    for food, restaurant, price in results:
        response_text += f"**{food}** at *{restaurant}* - **${price:.2f}**\n"

    response_text += "\nWould you like me to place an order for any of these dishes?"

    return response_text

def is_valid_phone_number(phone_number: str) -> bool:
    """
    Validates if the phone number follows the format: 555-666-7777.

    :param phone_number: The phone number entered by the user.
    :return: True if valid, False otherwise.
    """
    pattern = r"^\d{3}-\d{3}-\d{4}$"
    return bool(re.match(pattern, phone_number))

def order_management(action: str, order_id: str = None, phone_number: str = None,
                          person_name: str = None, comment: str = None):
    """
    Manages orders, including checking status, canceling orders, and adding comments.

    :param action: The action to perform ('check_status', 'cancel', 'add_comment').
    :param order_id: The order ID (required for all actions).
    :param phone_number: The user's phone number (required for canceling an order).
    :param person_name: The person's name (required for adding a comment).
    :param comment: The comment text (required for adding a comment).
    :return: Result message.
    """
    if action not in ["check_status", "cancel", "add_comment"]:
        return "Invalid action. Please use 'check_status', 'cancel', or 'add_comment'."

    if not order_id:
        return "Order ID is required for this action."

    if action == "cancel":
        if not phone_number:
            return "Phone number is required to cancel an order."

        if not is_valid_phone_number(phone_number):
            return (
                f"Invalid phone number format: '{phone_number}'.\n"
                "Please enter your phone number in the correct format: ***000-000-0000***."
            )

        connection = sqlite3.connect('/Users/erfanshahabi/Desktop/uni/NLP/NLP-CA6/Codes/food_orders.db')
        cursor = connection.cursor()
        cursor.execute("SELECT status FROM food_orders WHERE id = ? AND person_phone_number = ?", (order_id, phone_number))
        result = cursor.fetchone()

        if result is None:
            return f"Order ID {order_id} with phone number {phone_number} does not exist."

        if result[0] == "preparation":
            cursor.execute("UPDATE food_orders SET status = 'canceled' WHERE id = ?", (order_id,))
            connection.commit()
            connection.close()
            return f"Order ID {order_id} has been successfully canceled."
        else:
            connection.close()
            return f"Order ID {order_id} cannot be canceled as it is in '{result[0]}' status."

    elif action == "check_status":
        connection = sqlite3.connect('/Users/erfanshahabi/Desktop/uni/NLP/NLP-CA6/Codes/food_orders.db')
        cursor = connection.cursor()
        cursor.execute("SELECT status FROM food_orders WHERE id = ?", (order_id,))
        result = cursor.fetchone()
        connection.close()

        if result is None:
            return f"Order ID {order_id} does not exist."

        return f"Order ID {order_id} is currently in '{result[0]}' status."

    elif action == "add_comment":
        if not person_name or not comment:
            return "Both person name and comment are required to add a comment."

        connection = sqlite3.connect('/Users/erfanshahabi/Desktop/uni/NLP/NLP-CA6/Codes/food_orders.db')
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM food_orders WHERE id = ?", (order_id,))
        result = cursor.fetchone()

        if result is None:
            return f"Order ID {order_id} does not exist."

        cursor.execute("UPDATE food_orders SET comment = ? WHERE id = ?", (comment, order_id))
        connection.commit()
        connection.close()

        return f"Comment for Order ID {order_id} from {person_name} has been updated."

os.environ["TAVILY_API_KEY"] = "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"

def lancedb_search(query: str):
  """
  use the followed docements and generate answer for user query based on this documents.
  if the documents are not related to user querye retutn: not related
  Args:
      query: The query to search for
  """
  print("-----------Lancedb SEARCHING Tool----------")
  print("searching for", query)
  results = tbl.search(query).limit(5).to_pandas()
  print("results", results)
  return results


def web_search(query: str):
    """If the results of lancdb_search was not related, Search the web for the query.
    Args:
        query: The query to search for.
    """
    print("-----------SEARCHING Tool----------")
    print("searching for", query)
    results = TavilySearchResults( max_results=3).invoke(query)
    print("results", results)
    return results



def food_search(food_name=None, restaurant_name=None, max_distance=1):
    """
    Search for foods based on food_name, restaurant_name, or both using edit distance.
    :param connection: SQLite database connection
    :param food_name: Food name to search for (optional)
    :param restaurant_name: Restaurant name to search for (optional)
    :param max_distance: Maximum allowed edit distance for a match
    :return: List of matching foods
    """
    connection = sqlite3.connect('/content/drive/My Drive/nlp_ca6/food_orders.db')

    cursor = connection.cursor()
    cursor.execute("SELECT id, food_name, food_category, restaurant_name, price FROM foods")
    results = cursor.fetchall()

    matches = []
    for food_id, db_food_name, food_category, db_restaurant_name, db_price in results:
        food_name_distance = float('inf')
        restaurant_name_distance = float('inf')

        if food_name:
            food_name_distance_1 = Levenshtein.distance(food_name.lower(), db_food_name.lower(), weights=(0, 1, 1))
            food_name_distance_2 = Levenshtein.distance(food_name.lower(), db_food_name.lower(), weights=(1, 0, 1))
            food_name_distance_3 = Levenshtein.distance(food_name.lower(), db_food_name.lower(), weights=(1, 1, 1))
            food_name_distance = min(food_name_distance_1, food_name_distance_2, food_name_distance_3)


        if restaurant_name:
            restaurant_name_distance_1 = Levenshtein.distance(restaurant_name.lower(), db_restaurant_name.lower(), weights=(0, 1, 1))
            restaurant_name_distance_2 = Levenshtein.distance(restaurant_name.lower(), db_restaurant_name.lower(), weights=(1, 0, 1))
            restaurant_name_distance_3 = Levenshtein.distance(restaurant_name.lower(), db_restaurant_name.lower(), weights=(1, 1, 1))

            restaurant_name_distance = min(restaurant_name_distance_1, restaurant_name_distance_2, restaurant_name_distance_3)

        if food_name and restaurant_name:
            if food_name_distance <= max_distance and restaurant_name_distance <= max_distance:
                matches.append({
                    'id': food_id,
                    'food_name': db_food_name,
                    'food_category': food_category,
                    'restaurant_name': db_restaurant_name,
                    'price': db_price,
                    'edit_distance': min(food_name_distance, restaurant_name_distance)
                })
        elif food_name:
            if food_name_distance <= max_distance:
                matches.append({
                    'id': food_id,
                    'food_name': db_food_name,
                    'food_category': food_category,
                    'restaurant_name': db_restaurant_name,
                    'price': db_price,
                    'edit_distance': food_name_distance
                })
        elif restaurant_name:
            if restaurant_name_distance <= max_distance:
                matches.append({
                    'id': food_id,
                    'food_name': db_food_name,
                    'food_category': food_category,
                    'restaurant_name': db_restaurant_name,
                    'price': db_price,
                    'edit_distance': restaurant_name_distance
                })

    matches.sort(key=lambda x: x['edit_distance'])
    connection.close()
    return matches

tools = [food_search, order_management, food_recommendation, lancedb_search, web_search]
llm_with_tools = gpt4o_chat.bind_tools(tools)

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from typing import List, TypedDict
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from datetime import datetime
import json

sys_msg = SystemMessage(
    content="""You are a helpful assistant. Engage in a multi-turn conversation and assist the user with their queries.
    You have 5 tools: food_search, order_management, food_recommendation, lancedb_search, web_search.
    - If the user is looking for the availability of a food in a restaurant, use the food_search tool.
    - If the user is looking for general information about food, first use the lancedb_search tool. If the retrieved documents are not related, use the web_search tool. dont use both of them.
    - If the user wants a food recommendation but hasn't specified a food name, use the food_recommendation tool.
    - If the user wants order management, use the order_management tool.
    - Be careful, think step by step,if the user requests another operation during an ongoing order management operation, ask them to complete the previous operation first.
    """
)



class SummaryState(TypedDict):
    messages: List[HumanMessage]
    question: str
    answer: str
    summary: str

def reasoner(state: MessagesState):
    print("Assistant is thinking...")
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    print("Assistant:", response.content)
    return {"messages": [response]}

def summarize(state: MessagesState) -> MessagesState:
    summary_prompt = HumanMessage(content="""
    Summarize the above conversation while preserving full context, key points, and user intent.
    Your response should be concise yet informative.
    """)

    messages = state.get("messages", []) + [summary_prompt]
    response = llm_with_tools.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"messages": delete_messages + [response]}

builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_node("summarize", summarize)

builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "summarize")
builder.add_edge("summarize", "reasoner")
builder.add_edge("reasoner", END)

react_graph = builder.compile()
