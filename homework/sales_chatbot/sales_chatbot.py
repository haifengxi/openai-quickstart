import gradio as gr
import os

from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def initialize_sales_bot(vector_store_dirs):
    script_dir = os.path.dirname(__file__)
    print(f"script_dir: {script_dir}")
    print(f"vector_store_dirs: {vector_store_dirs}")

    global SALES_BOTS
    bots = []
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    for store_dir in vector_store_dirs:
        print(f"store_dir: {store_dir}")
        abs_dir_path = os.path.join(script_dir, store_dir)
        print(f"abs_dir_path: {abs_dir_path}")
        db = FAISS.load_local(abs_dir_path, OpenAIEmbeddings())
        bot = RetrievalQA.from_chain_type(
                    llm,
                    retriever=db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.8})
                )
        bot.return_source_documents = True
        bots.append(bot)

    SALES_BOTS = bots
    return SALES_BOTS

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    chosen_ans = None
    best_score = 0
    for bot in SALES_BOTS:
        ans = bot({"query": message})
        if len(ans["source_documents"]) > best_score:
            chosen_ans = ans
            best_score = len(ans["source_documents"])

    if chosen_ans is None:
        return "您这个问题我回答不了，请换个问题或问法"
    
    print(f"[result]{chosen_ans['result']}")
    print(f"[source_documents]{chosen_ans['source_documents']}")
    return chosen_ans["result"]    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="智能销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化智能销售机器人
    initialize_sales_bot(["real_estate_sales","electrical_appliances_sales"])
    # 启动 Gradio 服务
    launch_gradio()
