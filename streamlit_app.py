from typing import Callable, Any, Dict, List, Optional

# Ensure Streamlit components are properly imported
import streamlit as st
import os
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

# Embedding models
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# LLM models
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# Configure page
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False

# Helper Functions
@st.cache_resource
def get_embedding_model(provider: str = "ollama"):
    """Get embedding model based on provider choice."""
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif provider == "ollama":
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

@st.cache_resource
def get_llm(provider: str = "ollama", model: str = None):
    """Get LLM based on provider choice."""
    if provider == "openai":
        model = model or "gpt-3.5-turbo"
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=1000
        )
    elif provider == "ollama":
        model = model or "llama3.2:1b"
        return OllamaLLM(
            model=model,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

class RAGPipeline:
    def __init__(self, embedding_provider="ollama", llm_provider="ollama", llm_model=None):
        """Initialize RAG pipeline with flexible provider options."""
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        
        # Initialize embeddings
        try:
            self.embeddings = get_embedding_model(embedding_provider)
        except Exception as e:
            st.error(f"Failed to initialize embeddings ({embedding_provider}): {e}")
            self.embeddings = None
        
        # Initialize LLM
        try:
            self.llm = get_llm(llm_provider, llm_model)
        except Exception as e:
            st.error(f"Failed to initialize LLM ({llm_provider}): {e}")
            self.llm = None
        
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
    def create_vectorstore(self, documents, collection_name="rag_collection", persist_directory=None):
        """Create ChromaDB vector store from documents."""
        if not self.embeddings:
            raise ValueError("Embeddings not available")
            
        if persist_directory:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name
            )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return self.vectorstore
    
    def load_vectorstore(self, persist_directory, collection_name="ai_knowledge_base"):
        """Load an existing ChromaDB vector store from disk."""
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Vector store directory not found: {persist_directory}")
        
        if not self.embeddings:
            raise ValueError("Embeddings not available")
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return self.vectorstore
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        if not self.llm:
            raise ValueError("LLM not available")
        
        if not self.retriever:
            raise ValueError("Retriever not available")
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def query(self, question: str):
        """Query the RAG system."""
        if not self.qa_chain:
            raise ValueError("QA chain not configured")
        
        result = self.qa_chain({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    
    def update_llm(self, llm_provider: str, llm_model: str = None):
        """Update the LLM model dynamically."""
        try:
            self.llm = get_llm(llm_provider, llm_model)
            if self.retriever:
                self.setup_qa_chain()  # Reconfigure the QA chain with the new LLM
            st.success(f"LLM updated to {llm_provider} ({llm_model})")
        except Exception as e:
            st.error(f"Failed to update LLM ({llm_provider}): {e}")

def create_sample_documents():
    """Create sample documents for demonstration."""
    sample_documents = [
        """
        Machine Learning is a subset of artificial intelligence that enables computers to learn and improve 
        from experience without being explicitly programmed. It focuses on the development of computer programs 
        that can access data and use it to learn for themselves. The process of learning begins with observations 
        or data, such as examples, direct experience, or instruction, in order to look for patterns in data and 
        make better decisions in the future based on the examples that we provide.
        """,
        """
        Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") 
        to model and understand complex patterns in data. These neural networks attempt to simulate the behavior 
        of the human brain, allowing it to "learn" from large amounts of data. Deep learning has been particularly 
        successful in areas such as image recognition, natural language processing, and speech recognition.
        """,
        """
        Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction 
        between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, 
        understand, and make sense of human languages in a manner that is valuable. NLP combines computational 
        linguistics with statistical, machine learning, and deep learning models to help computers process human language.
        """,
        """
        Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of parametric and 
        non-parametric knowledge. It retrieves relevant information from a knowledge base and uses that information 
        to generate more accurate and contextually relevant responses. RAG models first retrieve relevant documents 
        from a corpus using a retriever, then use a generator to produce the final output conditioned on both the 
        query and the retrieved documents.
        """
    ]
    
    return [Document(page_content=doc.strip(), metadata={"source": f"doc_{i}"}) 
            for i, doc in enumerate(sample_documents)]

def load_webpage(url: str) -> Optional[Document]:
    """Load content from a webpage."""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        if documents:
            return documents[0]
        else:
            return None
    except Exception as e:
        st.error(f"Error loading {url}: {e}")
        return None

# Streamlit App UI
def main():
    st.title("🤖 RAG Chat Assistant")
    st.markdown("*Powered by LangChain, ChromaDB, and Ollama*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["ollama", "openai"],
            help="Choose your embedding provider. Ollama is free and local."
        )
        
        llm_provider = st.selectbox(
            "LLM Provider", 
            ["ollama", "openai"],
            help="Choose your language model provider."
        )
        
        if llm_provider == "ollama":
            llm_model = st.selectbox(
                "Ollama Model",
                ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
                help="Choose your Ollama model. Lighter models are faster."
            )
        else:
            llm_model = st.selectbox(
                "OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                help="Choose your OpenAI model."
            )
        
        # Track the selected LLM model in session state
        if "selected_llm_model" not in st.session_state:
            st.session_state.selected_llm_model = llm_model

        # Check if the selected model has changed
        if st.session_state.selected_llm_model != llm_model:
            st.session_state.selected_llm_model = llm_model
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.update_llm(llm_provider, llm_model)
        
        st.divider()
        
        # Vector Store Management
        st.header("📚 Knowledge Base")
        
        vector_store_option = st.radio(
            "Vector Store Option",
            ["Load Existing Vector Store"]
        )
        
        persist_directory = st.text_input(
            "Persist Directory",
            value="./rag_vectorstore",
            help="Directory to save/load vector store"
        )
        
        if vector_store_option == "Create New from Web":
            web_url = st.text_input(
                "Website URL", 
                value="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                help="URL to load content from"
            )
        
        # Initialize/Reset RAG System
        if st.button("🚀 Initialize RAG System", type="primary"):
            initialize_rag_system(
                embedding_provider, llm_provider, llm_model, 
                vector_store_option, persist_directory, 
                web_url if vector_store_option == "Create New from Web" else None
            )
        
        # System Status
        st.divider()
        st.header("📊 System Status")
        
        if st.session_state.rag_pipeline:
            st.success("✅ RAG Pipeline: Ready")
            if st.session_state.vectorstore_loaded:
                st.success("✅ Vector Store: Loaded")
                if st.session_state.rag_pipeline.vectorstore:
                    count = st.session_state.rag_pipeline.vectorstore._collection.count()
                    st.info(f"📄 Documents: {count}")
            else:
                st.warning("⚠️ Vector Store: Not loaded")
        else:
            st.error("❌ RAG Pipeline: Not initialized")
    
    # Main chat interface
    if st.session_state.rag_pipeline and st.session_state.vectorstore_loaded:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:** {source.metadata.get('source', 'unknown')}")
                            st.markdown(f"```\n{source.page_content[:200]}...\n```")
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about the knowledge base..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_pipeline.query(prompt)
                        
                        # Display answer
                        st.markdown(result["answer"])
                        
                        # Display sources
                        if result["sources"]:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(result["sources"], 1):
                                    st.markdown(f"**Source {i}:** {source.metadata.get('source', 'unknown')}")
                                    st.markdown(f"```\n{source.page_content[:200]}...\n```")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result["answer"],
                            "sources": result["sources"]
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    else:
        st.info("👈 Please initialize the RAG system using the sidebar to start chatting!")
        
        # Show sample questions
        st.subheader("💡 Sample Questions")
        st.markdown("""
        Once you initialize the system, you can ask questions like:
        - What is machine learning?
        - How does deep learning work?
        - Explain retrieval-augmented generation
        - What are the applications of NLP?
        """)

def initialize_rag_system(embedding_provider, llm_provider, llm_model, vector_store_option, persist_directory, web_url=None):
    """Initialize the RAG system with given parameters."""
    try:
        with st.spinner("Initializing RAG system..."):
            # Initialize RAG pipeline
            rag = RAGPipeline(
                embedding_provider=embedding_provider,
                llm_provider=llm_provider, 
                llm_model=llm_model
            )
            
            # Handle vector store based on option
            if vector_store_option == "Use Sample Documents":
                documents = create_sample_documents()
                
                # Chunk documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=100,
                    length_function=len,
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create vector store
                rag.create_vectorstore(
                    documents=chunks,
                    collection_name="sample_docs",
                    persist_directory=persist_directory
                )
                
            elif vector_store_option == "Load Existing Vector Store":
                if os.path.exists(persist_directory):
                    rag.load_vectorstore(
                        persist_directory=persist_directory,
                        collection_name="ai_knowledge_base"
                    )
                else:
                    st.error(f"Vector store directory not found: {persist_directory}")
                    return
                    
            elif vector_store_option == "Create New from Web" and web_url:
                web_document = load_webpage(web_url)
                if web_document:
                    # Chunk web document
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=400,
                        chunk_overlap=100,
                        length_function=len,
                    )
                    chunks = text_splitter.split_documents([web_document])
                    
                    # Create vector store
                    rag.create_vectorstore(
                        documents=chunks,
                        collection_name="web_docs",
                        persist_directory=persist_directory
                    )
                else:
                    st.error("Failed to load webpage content")
                    return
            
            # Setup QA chain
            rag.setup_qa_chain()
            
            # Update session state
            st.session_state.rag_pipeline = rag
            st.session_state.vectorstore_loaded = True
            st.session_state.messages = []  # Clear chat history
            
            st.success("🎉 RAG system initialized successfully!")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")

if __name__ == "__main__":
    main()
