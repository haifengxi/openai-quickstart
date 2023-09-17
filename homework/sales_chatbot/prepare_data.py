from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import os

def prepare_data():
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, "electrical_appliances_sales_qa.txt")
    with open(abs_file_path) as f:
        qa_data = f.read()

    text_splitter = CharacterTextSplitter(        
        separator = "\n\n",
        chunk_size = 150,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = False,
    )

    docs = text_splitter.create_documents([qa_data])

    db = FAISS.from_documents(docs, OpenAIEmbeddings())

    query = "这个电器防水吗"
    answer_list = db.similarity_search(query)
    for ans in answer_list:
        print(ans.page_content + "\n")

    db.save_local("electrical_appliances_sales")

if __name__ == "__main__":
    prepare_data()
