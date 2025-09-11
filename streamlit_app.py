from typing import Callable, Any, Dict, List, Optional

# Ensure Streamlit components are properly imported
import requests
import streamlit as st
import os
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader

# Embedding models
from langchain_community.embeddings import OllamaEmbeddings

# LLM models

from agent.tools import TOOLS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models.base import init_chat_model

# Configure page
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
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
    if provider == "ollama":
        return OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@st.cache_resource
def get_llm(provider: str = "ollama", model: str = None):
    """Get LLM based on provider choice."""
    if provider == "ollama":
        model = model or "llama3.2:1b"
        return init_chat_model(model=model, model_provider=provider)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class RAGPipeline:
    def __init__(
        self, embedding_provider="ollama", llm_provider="ollama", llm_model=None
    ):
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
        self.tools = TOOLS

    def create_vectorstore(
        self, documents, collection_name="default_collection", persist_directory=None
    ):
        """Create ChromaDB vector store from documents."""
        if not self.embeddings:
            raise ValueError("Embeddings not available")

        if persist_directory:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
            )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        return self.vectorstore

    def load_vectorstore(self, persist_directory, collection_name="default_collection"):
        """Load an existing ChromaDB vector store from disk."""
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(
                f"Vector store directory not found: {persist_directory}"
            )

        if not self.embeddings:
            raise ValueError("Embeddings not available")

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        return self.vectorstore

    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        if not self.llm:
            raise ValueError("LLM not available")

        if not self.retriever:
            raise ValueError("Retriever not available")

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = self.vectorstore.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        tools = [retrieve_context]+self.tools
        # If desired, specify custom instructions
        prompt = "You are a helpful assistant that provides information about RSECon25 based on the provided context."
        agent = create_agent(self.llm, tools, prompt=prompt)
        return agent

    def query(self, question: str):
        """Query the RAG system."""

        self.qa_chain = self.setup_qa_chain()

        response = self.qa_chain.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        source_docs_messages = [
            msg for msg in response["messages"] if msg.type == "tool"
        ]
        source_docs = [msg.artifact for msg in source_docs_messages][0]

        return {
            "question": question,
            "answer": response["messages"][-1].content,
            "sources": source_docs,
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

    import re

    def parse_model_key(name: str):
        """
        Split model name into (family, param_count).
        Non-numeric param counts are treated as float('inf') to push them last.
        """
        family, _, size = name.partition(":")
        if not size:
            return family, float("inf")

        # Extract numeric part and unit (b = billion, m = million)
        match = re.match(r"(\d+)([bm])?", size)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            if unit == "b":
                num *= 1_000_000_000
            elif unit == "m":
                num *= 1_000_000
            return family, num
        return family, float("inf")

    available_models = requests.get("http://localhost:11434/api/tags").json()
    models = [
        model["name"]
        for model in available_models["models"]
        if "embed" not in model["name"].lower()
    ]

    available_models = sorted(models, key=parse_model_key)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        llm_model = st.selectbox(
            "Ollama Model",
            available_models,
            help="Choose your Ollama model. Lighter models are faster.",
        )

        # Track the selected LLM model in session state
        if "selected_llm_model" not in st.session_state:
            st.session_state.selected_llm_model = llm_model

        # Check if the selected model has changed
        if st.session_state.selected_llm_model != llm_model:
            st.session_state.selected_llm_model = llm_model
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.update_llm(llm_model)

        st.divider()

        # Vector Store Management
        st.header("📚 Knowledge Base")

        vector_store_option = "Load Existing Vector Store"

        # Initialize/Reset RAG System
        if st.button("🚀 Initialize RAG System", type="primary"):
            initialize_rag_system(
                "ollama",
                "ollama",
                llm_model,
                vector_store_option,
                "./vectorstore",
                None,
            )

        # System Status
        st.divider()
        st.header("📊 System Status")

        if st.session_state.rag_pipeline:
            st.success("✅ RAG Pipeline: Ready")
            if st.session_state.vectorstore_loaded:
                st.success("✅ Vector Store: Loaded")
                if st.session_state.rag_pipeline.vectorstore:
                    count = (
                        st.session_state.rag_pipeline.vectorstore._collection.count()
                    )
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
                            st.markdown(
                                f"**Source {i}:** {source.metadata.get('source', 'unknown')}"
                            )
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
                    result = st.session_state.rag_pipeline.query(prompt)

                    # Display answer
                    st.markdown(result["answer"])

                    # Display sources
                    if result["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(
                                    f"**Source {i}:** {source.metadata.get('source', 'unknown')}"
                                )
                                st.markdown(f"```\n{source.page_content[:200]}...\n```")

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info(
            "👈 Please initialize the RAG system using the sidebar to start chatting!"
        )

        # Show sample questions
        st.subheader("💡 Sample Questions")
        st.markdown(
            """
        Once you initialize the system, you can ask questions like:
        - What is machine learning?
        - How does deep learning work?
        - Explain retrieval-augmented generation
        - What are the applications of NLP?
        """
        )


def initialize_rag_system(
    embedding_provider,
    llm_provider,
    llm_model,
    vector_store_option,
    persist_directory,
    web_url=None,
):
    """Initialize the RAG system with given parameters."""
    try:
        with st.spinner("Initializing RAG system..."):
            # Initialize RAG pipeline
            rag = RAGPipeline(
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )

            # Handle vector store based on option
            if vector_store_option == "Load Existing Vector Store":
                if os.path.exists(persist_directory):
                    rag.load_vectorstore(
                        persist_directory=persist_directory,
                        collection_name="default_collection",
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
                        collection_name="default_collection",
                        persist_directory=persist_directory,
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
