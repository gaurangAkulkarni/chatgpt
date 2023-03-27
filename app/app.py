from flask import Flask, request
import os
import sys

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex
from llama_index import LLMPredictor, PromptHelper

from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)

## Chat tooling

# llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024)
llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1024)
memory = ConversationBufferWindowMemory(k=2, memory_key='chat_history', return_messages=True)

constitution_index = GPTSimpleVectorIndex.load_from_disk('./app/indexes/01Constitution.json')
lubus_index = GPTSimpleVectorIndex.load_from_disk('./app/indexes/02Lubus.json')
covid_index = GPTSimpleVectorIndex.load_from_disk('./app/indexes/03Covid.json')

tools = [
    Tool(
        name="constitution_index",
        func=lambda q: constitution_index.query(q),
        description=f"Useful when you want answer questions about the Constitution.",
    ),
    Tool(
        name="lubus_index",
        func=lambda q: lubus_index.query(q),
        description=f"Useful when you want answer questions about the Lubus.",
    ),
    Tool(
        name="covid_index",
        func=lambda q: covid_index.query(q),
        description=f"Useful when you want answer questions about the Covid.",
    ),
]

agent_chain = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False, memory=memory,  max_iterations=6, early_stopping_method="generate")

def fetch_query_data_from_openai(chat_query):  # this is the home page function that generates the page code
    result = agent_chain.run(input=chat_query)
    print(result)
    # result = result.replace('"', '\\"').replace('\n', '\\n')
    return result

@app.route('/')  # this is the home page route
def hello_world():  # this is the home page function that generates the page code
    return "Hello world!"

@app.route('/chatbot') 
def openai_api_call():  # this is the home page function that generates the page code
    chat_query = request.args.get('chat_query')
    result = fetch_query_data_from_openai(chat_query)
    return {
        "fulfillmentText":
        result,
        "source":
        "webhookdata"
    }
    return '200'

@app.route('/webhook', methods=['POST'])
async def webhook():
    try:
        req = request.get_json(silent=True, force=True)
        chat_query = req.get('chat_query')
        result = fetch_query_data_from_openai(chat_query)

        return {
            "fulfillmentText":
            result,
            "source":
            "webhookdata"
        }
        return '200'
    except Exception as e:
        print('error',e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('oops',exc_type, fname, exc_tb.tb_lineno)
        return '400'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", default=5001))