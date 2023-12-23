from utils import *
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

st.title("GitHub Repo ChatBot")

git_repo_link = st.text_input("Put your GitHub Repo Link")
repo_path,chroma_path = cloning(git_repo_link)
docs = extract_all_files(repo_path)
texts = chunk_files(docs)
st.write("Cloning Complete")
st.write("Creating Embeddings...")
embeddings = create_embeddings(texts)
vectordb = load_db(texts, embeddings,repo_path,chroma_path)
st.write("Embedding Complete")

if "messages" not in st.session_state:
            st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your question here."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    response,metadata = retrieve_results(prompt,vectordb)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(metadata)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})