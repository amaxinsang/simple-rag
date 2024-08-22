#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import uuid
import glob
import pickle

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings


# In[ ]:


class DocumentManager:
    def __init__(self, directory_path, glob_pattern="./*.md"):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.documents = []
        self.all_sections = []
    
    def load_documents(self):
        loader = DirectoryLoader(
            self.directory_path, 
            glob=self.glob_pattern, 
            show_progress=True, 
            loader_cls=UnstructuredMarkdownLoader
            )
        self.documents = loader.load()
    

    def split_documents(self):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        for doc in self.documents:
            sections = text_splitter.split_text(doc.page_content)
            self.all_sections.extend(sections)


# In[ ]:


def save_to_pickle(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# In[ ]:


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# In[ ]:


model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


# In[ ]:


doc_manager = DocumentManager('./data/sagemaker_documentation')
doc_manager.load_documents()
doc_manager.split_documents()


# In[ ]:


documents = doc_manager.documents

for index, doc in tqdm(enumerate(documents)):

    summarize_prompt = """Summarize the following document:\n\n{doc}"""
    summarize_prompt = summarize_prompt.format(doc=doc.page_content)
    messages = [
        {"role": "user", "content": summarize_prompt}
    ]

    outputs = pipe(messages, max_new_tokens=256, do_sample=True)
    summary = outputs[0]["generated_text"][-1]["content"].strip()

    with open(f"./data/processed/summary_{index:03d}.md", "a+") as file:
        file.write(summary)


# In[ ]:


summaries = []
for filename in sorted(glob.glob("./data/processed/*")):
    with open(filename, "r") as file:
        summary = file.read()
        summaries.append(summary)


# In[ ]:


embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device':device}
)

vectorstore = Chroma(collection_name="summaries",
                     embedding_function=embedder,
                     persist_directory="./chroma_langchain_db")

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)


# In[ ]:


doc_ids = [str(uuid.uuid4()) for _ in doc_manager.documents] 

summary_docs = [
    Document(page_content=summary, metadata={id_key: doc_ids[index]})
    for index, summary in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, doc_manager.documents)))
save_to_pickle(retriever.byte_store.store, "./docstore.pkl")


# In[ ]:




