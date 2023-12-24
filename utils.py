from git import Repo
import os
import torch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

allowed_extensions = ['.py', '.ipynb', '.md']
model_name = "all-MiniLM-L6-v2"
model_kwargs={'device': 'cpu'}

def cloning(url):
    current_path = os.getcwd()
    last_name = url.split('/')[-1]
    clone_path = last_name.split('.')[0]
    repo_path = os.path.join(current_path,clone_path)
    chroma_path = f'{clone_path}-chroma'

    if not os.path.exists(repo_path):
        Repo.clone_from(url, to_path=repo_path)
    return repo_path,chroma_path

def extract_all_files(repo_path):
        root_dir = repo_path
        docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try: 
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        docs.extend(loader.load_and_split())
                    except Exception as e:
                        pass
        return docs

def chunk_files(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)
        num_texts = len(texts)
        return texts

def create_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(model_name= model_name,model_kwargs=model_kwargs)
    return embeddings


def load_db(texts, embeddings,repo_path,chroma_path):
    if os.path.exists(chroma_path):
         vectordb = Chroma(embedding_function=embeddings, persist_directory=chroma_path)
    else:
        vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=chroma_path)
        vectordb.persist()
    return vectordb

def retrieve_results(query,vectordb):
        llm ='meta-llama/Llama-2-7b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(llm)


        model = AutoModelForCausalLM.from_pretrained(llm,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             load_in_8bit=True,
                                              #load_in_4bit=True
                                             )
        pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 1024,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
        
        llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
        chain =  RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff",return_source_documents=True, retriever=vectordb.as_retriever())
        result=chain({"query": query}, return_only_outputs=True)
        return result['answer'], result['source_documents']


