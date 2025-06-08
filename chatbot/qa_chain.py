from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from chatbot.vector_store import load_vector_store

def get_jobconnect_qa_chain():
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
