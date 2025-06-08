from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from chatbot.vector_store import load_vector_store

def get_safe_jobconnect_qa_chain():
    # Load vector store
    vectorstore = load_vector_store()

    # Use your fine-tuned model
    llm = ChatOpenAI(
        temperature=0.3,
        model="ft:gpt-3.5-turbo-0125:personal::BS6z6ywt"
    )

    # Prompt template for consistent style
    prompt = PromptTemplate.from_template("""
You are a helpful assistant for the JobConnect system. Only answer questions related to the system (features, MBTI, resume, job matching, support). Do not answer questions unrelated to JobConnect.

Context:
{context}

Question:
{question}
""")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return chain
