#SUBMITTED BY ASHUTOSH JHA
#MANIPAL INSTITUTE OF TECHNOLOGY

# openvino-rag-client.py

import os
from dotenv import load_dotenv
import json
import requests
import streamlit as st

# Load environment variables
load_dotenv(verbose=True)
server_url = os.environ['SERVER_URL']
server_port = os.environ['SERVER_PORT']

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #17C7C7;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Title and logo
# Creating columns
col1, col2, col3 = st.columns(3)

# Adding images to columns
with col1:
    st.image('logo1.png', width=150)

with col2:
    st.image('logo2.png', width=150)

with col3:
    st.image('logo3.jpeg', width=150)
#st.image('logo.png', width=100)
st.title('PDF based Q&A Chatbot using RAG')


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input
prompt = st.chat_input('Your input here.')

if prompt:
    # Display user message
    with st.chat_message('user'):
        st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Get assistant response
    with st.chat_message('assistant'):
        payload = {"query": prompt}
        ans = requests.get(f'http://{server_url}:{server_port}/chatbot/1', params=payload)
        ans = json.loads(ans.text)
        st.markdown(ans['response'])
        st.session_state.messages.append({'role': 'assistant', 'content': ans['response']})
