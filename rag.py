from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


def create_retriever():

    files = [
        "knowledge/world.txt",
        "knowledge/characters.txt",
        "knowledge/rules.txt"
    ]

    documents = []

    for file in files:
        loader = TextLoader(file)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )