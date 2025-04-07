from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from langchain_chroma import Chroma
from typing import List, Any, Literal, Callable
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
import os

class DocumentSemanticChunker:
    def __init__(self):
        pass
    
    def init_chunker(self, embedding_function):
        self.semantic_chunker = SemanticChunker(embedding_function)
    
    def split_documents(self, documents):
        chunked_documents = self.semantic_chunker.create_documents([d.page_content for d in documents])
        return chunked_documents

class DocumentTextSplitChunker:
    def __init__(self, chunk_size : int, chunk_overlap : int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def split_documents(self, documents):
        chunked_documents = self.text_splitter.split_documents(documents)
        return chunked_documents


class ChromaDB:
    """
    Handles the ChromaDB instance.
    """
    _default_chroma_collection_name_ = 'collection'

    def __init__(self, embedding_function : Literal['OpenAI', 'HuggingFace', 'other'],
                text_chunker : DocumentSemanticChunker | DocumentTextSplitChunker, 
                 **kwargs):
        if embedding_function == 'HuggingFace':
            self.embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        elif embedding_function == 'OpenAI':
            self.embedding_function = OpenAIEmbeddings(openai_api_key=kwargs["openai_key"], model='text-embedding-3-large')
        else:
            self.embedding_function = kwargs["embedding_function"]
        self.text_chunker = text_chunker
        if text_chunker is DocumentSemanticChunker:
            self.text_chunker.init_chunker(self.embedding_function)

    def load_from_disk(self, at_path : str, collection : str = _default_chroma_collection_name_):
        self.disk_path = at_path
        self.db = Chroma(persist_directory=at_path, embedding_function=self.embedding_function, 
                        #  client_settings = Settings(allow_reset=True),
                         collection_name=collection)
    
    def load_http_client(self, host : str = 'localhost', port : int = 8000):
        self.db = Chroma(host=host, port=port, client_settings = Settings(allow_reset=True))

    def create_collection(self, save_path, source_path,
                          status_callback : Callable[[str, int], Any],
                          document_preparer : None | Callable[Document, Document],
                          collection = _default_chroma_collection_name_,
                          **kwargs):
        """
        :param save_path: Directory to save chromadb
        :param source_path: Source pdfs to read from.
        :param status_callback: Callback function to process status
        :param text_chunker_type: if semantic, semantic chunking via openAI will be done on documents, if text_split, use chunk_size and chunk_overlap kwargs parameters
        """
        self.collection = collection
        docs = self.load_chunk_persist_pdf(source_path, self.text_chunker, status_callback)
        if len(docs) > 0:
            status_callback("Embedding documents...", -1)
            self.db = Chroma.from_documents(documents=[document_preparer(doc) if document_preparer is not None else doc for doc in docs], 
                                            embedding=self.embedding_function, persist_directory=save_path,
                                            collection_name = collection)
            status_callback("Documents added. Database ready.", -1)
        else:
            status_callback("No documents to embed.", -1)

    def add_documents(self, abs_path : str, status_callback : Callable[[str, int], Any],
                          document_preparer : None | Callable[Document, Document]):
        docs = self.load_chunk_persist_pdf(abs_path, self.text_chunker, status_callback)
        status_callback("Embedding documents...", -1)
        if len(docs) > 0:
            self.db.add_documents([document_preparer(doc) if document_preparer is not None else doc for doc in docs])
            status_callback("Documents added. Database ready.", -1)
        else:
            status_callback("No documents to embed.", -1)


    def reset(self):
        self.db.delete_collection()
        self.db = Chroma(persist_directory=self.disk_path, embedding_function=self.embedding_function, client_settings = Settings(allow_reset=True),
                         collection_name=self.collection)

    
    def load_chunk_persist_pdf(self, abs_path : str, text_chunker, status_callback : Callable[[str, int], Any]) -> List[Any]:
        """
        Returns split chunks from documents:
        :param abs_path: path to extract documents from (a pdf file or folder)
        :param status_callback: called every time a new path is loaded. Receives as parameters (message: str, file_index: int)
        """
        # abs_path = os.path.join(os.getcwd(), relative_path)
        documents = []
        if abs_path.endswith('.pdf'):
            loader = PyPDFLoader(abs_path)
            documents.extend(loader.load())
            status_callback("'.pdf' file prepared", 0)
        else:
            index = 0
            for root, _, files in os.walk(abs_path):
                for file in files:
                    if file.endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        loader = PyPDFLoader(pdf_path)
                        documents.extend(loader.load())
                        status_callback(f"{file} - pdf file no. {index} prepared...", index)
                        index += 1
        chunked_documents = text_chunker.split_documents(documents)
        return chunked_documents
    
    def similarity_search(self, query):
        return self.similarity_search(query)

    def as_retriever(self, k, search_type):
        return self.db.as_retriever(k=k, search_type=search_type)