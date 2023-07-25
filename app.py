import streamlit as st
from streamlit_chat import message
import os
import pickle
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_csv_agent
from langchain.memory import ConversationBufferMemory
import yaml
from langchain.vectorstores import FAISS, Chroma,Pinecone
from langchain import PromptTemplate


from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


st.set_page_config(page_title="NN AI FAQ",page_icon = "drop_of_blood",)


col1, col2, col3 = st.columns(3)
with col1:  
    st.image("ChatbotImage.png",width = 600,output_format='PNG')

def prompt_template():
    prompt_template = """Use the following pieces of context to answer the questions about NN at the end.
{context}
Question: {question}
If you don't know the answer, just say that you don't know, don't try to make up an answer. Only give answers about the provided context. Answer in Dutch and then in a seperate paragraph repeat the same answer in English.
Helpful Answer:"""

    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


@st.cache_resource



def createChain(): 
    vectordb = Chroma(persist_directory='db', embedding_function=OpenAIEmbeddings())

    # @st.cache_resource
    # chain = ConversationalRetrievalChain.from_llm(
    # llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),as_retrieffver

    
    chain_type_kwargs = {"prompt": prompt_template()}
    chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'), chain_type="stuff", retriever=vectordb.as_retriever(),chain_type_kwargs = chain_type_kwargs)
    return chain
chain = createChain()

def conversational_chat(query):
    print(query)
    # result = chain({"question": query, 
    # "chat_history": st.session_state['history']})
    # st.session_state['history'].append((query, result["answer"]))
    result = chain.run(query)
    return result
with st.form(key="form1"):
    user_input = st.text_input("Query:", placeholder="Ask your question here!", key='input')
    submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input:
        output = conversational_chat(query=user_input)
        st.write(output)

# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = ["Hallo, stel me een vraag over het arbeidsreglement."]

# if 'past' not in st.session_state:
#     st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
# response_container = st.container()
#container for the user's text input
# container = st.container()

# with container:
#         with st.form(key='my_form', clear_on_submit=True):
#             # language_input = st.selectbox(label="language", options= ["English","Dari","Dutch","Ukranian"])
#             user_input = st.text_input("Query:", placeholder="Ask your question here!", key='input')
#             submit_button = st.form_submit_button(label='Send')

            
#         if submit_button and user_input:
            


#             output = conversational_chat(user_input)
            
#             st.session_state['past'].append(user_input)
#             st.session_state['generated'].append(output)
            
# if st.session_state['generated']:
#     with response_container:
#         for i in range(len(st.session_state['generated'])):
#             message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
#             message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
