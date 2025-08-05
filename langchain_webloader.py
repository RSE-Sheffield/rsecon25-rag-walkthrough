#%%
# LangChain Web Loader Example
# This example demonstrates how to use LangChain to load websites and extract HTML content

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests

# Basic Web Loader Example
def load_website_basic(url):
    """
    Load a website using LangChain's WebBaseLoader
    """
    try:
        # Create a web loader
        loader = WebBaseLoader(url)
        
        # Load the documents
        documents = loader.load()
        
        print(f"Loaded {len(documents)} documents from {url}")
        
        # Print basic info about the loaded content
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"Content length: {len(doc.page_content)} characters")
            print(f"Metadata: {doc.metadata}")
            print(f"First 200 characters: {doc.page_content[:200]}...")
            
        return documents
        
    except Exception as e:
        print(f"Error loading website: {e}")
        return None

# Advanced Web Loader with Custom Parsing
def load_website_advanced(url, css_selector=None):
    """
    Load a website with custom BeautifulSoup parsing
    """
    try:
        # Custom parsing function
        def custom_parser(content):
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract specific elements if CSS selector is provided
            if css_selector:
                elements = soup.select(css_selector)
                text_content = " ".join([elem.get_text() for elem in elements])
            else:
                text_content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        
        # Create loader with custom parser
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": None}  # Parse entire document
        )
        
        # Override the default parsing
        original_load = loader.load
        def custom_load():
            response = requests.get(url)
            response.raise_for_status()
            parsed_content = custom_parser(response.content)
            
            from langchain.schema import Document
            return [Document(
                page_content=parsed_content,
                metadata={"source": url, "title": BeautifulSoup(response.content, 'html.parser').title.string if BeautifulSoup(response.content, 'html.parser').title else ""}
            )]
        
        loader.load = custom_load
        documents = loader.load()
        
        return documents
        
    except Exception as e:
        print(f"Error loading website with advanced parser: {e}")
        return None

# Multiple URLs Loader
def load_multiple_websites(urls):
    """
    Load multiple websites at once
    """
    try:
        # Create loader with multiple URLs
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} documents from {len(urls)} URLs")
        
        # Group documents by source
        by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(doc)
        
        for source, docs in by_source.items():
            print(f"\n{source}: {len(docs)} documents")
            
        return documents
        
    except Exception as e:
        print(f"Error loading multiple websites: {e}")
        return None

# Web Loader with Text Splitting
def load_and_split_website(url, chunk_size=1000, chunk_overlap=200):
    """
    Load a website and split the content into chunks for RAG
    """
    try:
        # Load the website
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Split the documents
        split_docs = text_splitter.split_documents(documents)
        
        print(f"Original documents: {len(documents)}")
        print(f"Split into {len(split_docs)} chunks")
        
        # Show some statistics
        chunk_lengths = [len(doc.page_content) for doc in split_docs]
        print(f"Average chunk size: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
        print(f"Min chunk size: {min(chunk_lengths)} characters")
        print(f"Max chunk size: {max(chunk_lengths)} characters")
        
        return split_docs
        
    except Exception as e:
        print(f"Error loading and splitting website: {e}")
        return None

# Raw HTML Extraction
def get_raw_html(url):
    """
    Get raw HTML content from a URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print(f"Successfully fetched HTML from {url}")
        print(f"Content length: {len(response.content)} bytes")
        print(f"Content type: {response.headers.get('content-type', 'unknown')}")
        
        return response.text
        
    except Exception as e:
        print(f"Error fetching raw HTML: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example URLs
    test_url = "https://docs.python.org/3/tutorial/introduction.html"
    test_urls = [
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/controlflow.html"
    ]
    
    print("=== Basic Web Loading ===")
    docs = load_website_basic(test_url)
    
    print("\n=== Advanced Web Loading ===")
    advanced_docs = load_website_advanced(test_url, css_selector="p")
    
    print("\n=== Multiple URLs Loading ===")
    multi_docs = load_multiple_websites(test_urls)
    
    print("\n=== Load and Split for RAG ===")
    split_docs = load_and_split_website(test_url, chunk_size=500, chunk_overlap=50)
    
    print("\n=== Raw HTML Extraction ===")
    raw_html = get_raw_html(test_url)
    if raw_html:
        print(f"Raw HTML preview: {raw_html[:200]}...")

# Installation requirements:
# pip install langchain langchain-community beautifulsoup4 requests

print("\nTo use this code, make sure you have installed:")
print("pip install langchain langchain-community beautifulsoup4 requests")
#%%

# %%
