# import required libraries
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableSequence
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

# load .env file having openai_api_key
load_dotenv()
# model gpt 3.5 turbo
model = ChatOpenAI(model='gpt-3.5-turbo')
# embeddings
embedding = OpenAIEmbeddings()
# page title and header set
st.set_page_config("PDF Q&A")
st.header("PDF Q&A")
# structuring the output
class llm_answer(BaseModel):
    answer: str = Field(description="Answer to the query asked by the user, if not known reply in polite manner")
# pydantic parser
parser = PydanticOutputParser(pydantic_object=llm_answer)
# file uploader to be used by users
uploaded_file = st.file_uploader("Upload your pdf files here:", type='pdf')
# if the file has been uploaded
if uploaded_file is not None:
    # creating a new pdf file from the user uploaded file
    with open("user_file.pdf",'wb') as f:
        f.write(uploaded_file.getbuffer())
    # loading the pdf using document loader
    loader = PyPDFLoader("user_file.pdf")
    documents = loader.lazy_load()
    # splitting the document into multiple chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    # formatting docs to get only page_content & metadata
    formatted_docs = [
        Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs
    ]
    # faiss vector store
    vector_db = FAISS.from_documents(embedding=embedding, documents=formatted_docs)
    # prompt having question, context and the instructions from the parser
    prompt = PromptTemplate(
        template="Answer the question {question} based on the following: {context} \n {format_instruct}",
        input_variables=['question','context'],
        partial_variables={"format_instruct":parser.get_format_instructions()}
    )
    # Runnable sequence chain
    chain = prompt | model | parser
    # text input from the user as query
    query = st.text_input("Please enter your query here: ")
    if query:
        # searching vector store for similar 2 documents
        context = vector_db.similarity_search(query, k=2)
        # chain.invoke with query and context to llm
        result = chain.invoke({"question":query, "context":context})
        # showing the result in streamlit
        st.write(result.answer)
