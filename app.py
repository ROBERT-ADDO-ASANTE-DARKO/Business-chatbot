from pyexpat import model
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

from ingest import DB_FAISS_PATH

# Add custom CSS file to Chainlit's static folder
#app.config['EXTERNAL_STYLESHEETS'].append('/static/css/custom.css')

# Load the embeddings and vector store

#DB_FAISS_PATH = "vectorstores/db_fiass"
DB_FAISS_PATH = "vectorstores\db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    # Load pretrained language model
    llm = CTransformers(model = "TheBloke/Llama-2-7B-Chat-GGML",
                        model_type = "llama",
                        max_new_tokens = 1012,
                        temperature = 0.5)
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
    print(embeddings)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query', query})
    return response

# Chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot....")
    await msg.send()
    msg.content = "Hi, Welcome to KYC Nations, The Ultimate Real Estate Bot. What can we assist you with?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    #if sources:
    #    answer += f"\nSources:" + str(sources)
    #else:
    #    answer += "\nNo source documents found."
    
    #await cl.Message(content=answer).send()

#if __name__ == "__main__":
    # Set custom prompt