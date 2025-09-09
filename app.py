import streamlit as st
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# with st.chat_message(name="assistant"):
#     st.write("👋🏾 Hello there! How can I assist you today?")

st.title("Triage Agent")
with st.chat_message(name="assistant"):
    st.write("👋🏾 Hello there! How can I assist you today?")


# Load API key from .env file
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
print("GITHUB_TOKEN loaded:", repr(GITHUB_TOKEN)) 

client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

def call_github_llm(prompt, max_tokens=2048):
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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in call_github_llm(prompt):
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