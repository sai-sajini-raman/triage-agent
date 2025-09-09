import streamlit as st
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

import pandas as pd

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document


st.title("Triage Agent")
with st.chat_message(name="assistant"):
    st.write("👋🏾 Hello there! How can I assist you today?")


# Load API key from .env file
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
# print("GITHUB_TOKEN loaded:", repr(GITHUB_TOKEN)) 

# --- load docs & build index, including excel
@st.cache_resource
def build_index():
    docs = SimpleDirectoryReader("docs").load_data()
    excel_docs = []
    excel_path = "MAL-Food-SC.xlsx"
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        excel_docs = [
            Document(text=row.to_json())
            for _, row in df.iterrows()
        ]
    all_docs = docs + excel_docs

    # Create the embedding model using HuggingFace
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Pass the embed_model to the index
    index = VectorStoreIndex.from_documents(all_docs, embed_model=embed_model)
    return index
index = build_index()

# --- query llamaindex & get context
def get_relevant_context(query):
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    context = "\n".join([node.text for node in nodes])
    print("Retrieved context for query:", query)
    print(context)
    return context

client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# --- calling with context
def call_github_llm(prompt, context="", max_tokens=100):
    rag_prompt = f" Here is the data given as Context:\n{context}\nUse the above context to answer the following question. Strictly rely on only the given context data to answer the question. \n\nQuestion: {prompt}"
    response = client.complete(
        # messages=[UserMessage(prompt)],
        messages=[
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages
        ],
        stream=True,
        model="openai/gpt-4.1",
        max_tokens=max_tokens,
    )
    for chunk in response:
        yield chunk


# Initialize chat history
if "messages" not in st.session_state:   # messages --> session variable
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt:= st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": "You are a helpful assistant, who helps to triage incidents within an organization. You should ask relevant questions to gather information about the incident, such as its severity, impact, affected systems, etc,. Based on the information provided, you should categorize the incident and assign a priority level (e.g., low, medium, high) to indicate the urgency of the incident. Finally, you should suggest appropriate next steps for resolving the incident, such as escalating it to a higher-level support team or providing specific troubleshooting instructions, etc,."})
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- RAG: Retrieve context from docs,excel
    context = get_relevant_context(prompt)
   

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in call_github_llm(prompt, context):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        message_placeholder.markdown(full_response)





# if __name__ == "__main__":
#     prompt = 
#     output = call_github_llm()
#     print(output)