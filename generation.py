#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from typing import List, Dict

import torch
 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

import gradio as gr


# In[2]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[3]:


def load_from_pickle(filename: str):
    with open(filename, "rb") as file:
        return pickle.load(file)


# In[4]:


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


# In[5]:


embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': device}
)


# In[6]:


vectorstore = Chroma(collection_name="summaries",
                     embedding_function=embedder,
                     persist_directory="./chroma_langchain_db")
store = InMemoryByteStore()
id_key = "doc_id"

store_dict = load_from_pickle("./docstore.pkl")
store.mset(list(store_dict.items()))

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)


# In[7]:


def prompt_formatter(query: str, 
                     context_items: List[Dict]) -> str:
    
    context = "\n- " + "\n- ".join([item.replace("\n\n", "\n") for item in context_items])

    base_prompt = """Based on the following context items, please answer the query. 
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
\nNow use the following context items to answer the user query: {context}
\n\nUser query: {query}
Answer:
    """
    base_prompt = base_prompt.format(context=context, query=query)

    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt


# In[12]:


def ask(query, history):
        
    retrieved_docs = retriever.get_relevant_documents(query)

    context_items = [doc.page_content for doc in retrieved_docs]

    prompt = prompt_formatter(query=query,
                            context_items=context_items)
    
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**input_ids,
                            temperature=0.7, 
                            do_sample=True, 
                            max_new_tokens=256) 

    output_text = tokenizer.decode(outputs[0]).replace(prompt, '').replace("<bos>", "").replace("<eos>", "")

    return output_text


# In[13]:


gr.ChatInterface(ask).launch()


# In[ ]:





# In[ ]:





# In[ ]:




