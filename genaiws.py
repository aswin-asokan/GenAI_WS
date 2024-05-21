# -*- coding: utf-8 -*-

!pip -q install langchain
!pip install -U langchain-community
!pip -q install transformers
!pip -q install bitsandbytes accelerate xformers einops
!pip -q install datasets loralib sentencepiece
!pip -q install pypdf
!pip -q install docx2txt
!pip -q install sentence_transformers
!pip install chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from google.colab import files
from langchain import HuggingFacePipeline

import os
import subprocess
import sys
import torch
import io
import transformers
import tempfile
from torch import bfloat16

subprocess.run(["huggingface-cli", "login", "--token", "{{API_TOKEN}}"])

model_id = "Trendyol/Trendyol-LLM-7b-chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True)

uploaded = files.upload()

document = []

temp_dir = tempfile.TemporaryDirectory()

for filename, content in uploaded.items():
    file_path = os.path.join(temp_dir.name, filename)
    with open(file_path, 'wb') as f:
        f.write(content)

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        document.extend(loader.load())
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        document.extend(loader.load())
    elif filename.endswith('.txt'):
        loader = TextLoader(file_path)
        document.extend(loader.load())


temp_dir.cleanup()

document_splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=100)
document_chunks = document_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./data')
vectordb.persist()

generator = transformers.pipeline(
    model = model,
    tokenizer=tokenizer,
    return_full_text = True,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=512,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=generator)

pre_prompt = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nGenerate the next agent response by answering the question. Answer it as succinctly as possible. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities in your answer and use the titles to separate between topics or domains. If you cannot answer the question from the given documents, please state that you do not have an answer.\n"""
prompt = pre_prompt + "CONTEXT:\n\n{context}\n" +"Question : {question}" + "[\INST]"
llama_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), combine_docs_chain_kwargs={"prompt": llama_prompt}, return_source_documents=True)

chat_history = []

while True:
    query = input("Please enter your question (type 'exit' to quit): ")

    if query.lower() == 'exit':
        print("Exiting...")
        break

    result = chain({"question": query, "chat_history": chat_history})

    print("Answer:", result['answer'])
    chat_history.append((query, result["answer"]))
