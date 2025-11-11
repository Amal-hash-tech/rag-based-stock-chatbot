import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: Groq API key not found in .env")
    exit(1)

# Initialize the LLaMA 3.3 70B model via Groq
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024
)

# Initialize embeddings (using free HuggingFace embeddings)
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Define the system prompt for RAG-based stock exchange chatbot
system_prompt = """You are an expert financial advisor specializing in stock exchanges and equity markets.

Use the following context from the knowledge base to answer the user's question. 
If the context contains relevant information, use it to provide a detailed answer.
If the context doesn't contain relevant information, use your general knowledge about stock exchanges.

Always cite when you're using information from the provided context.

Context from knowledge base:
{context}

Your knowledge covers:
- Stock market operations and mechanics
- Trading strategies and investment principles
- Market indices (S&P 500, NASDAQ, DOW, etc.)
- Stock valuation and analysis
- Market trends and economic indicators
- Risk management and portfolio diversification
- IPOs, dividends, and corporate actions

Provide clear, accurate, and educational responses about stock exchange topics.
Remember previous parts of our conversation and reference them when relevant."""

# Simple window-based chat history
class WindowChatMessageHistory:
    """Chat message history that keeps only the last K messages"""
    def __init__(self, k: int = 10):
        self.messages = []
        self.k = k
    
    def add_message(self, message):
        """Add a message and maintain window size"""
        self.messages.append(message)
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k:]
    
    def clear(self):
        """Clear all messages"""
        self.messages = []

# Initialize chat history
chat_history = WindowChatMessageHistory(k=10)

# Global variable for vector store
vectorstore = None
retriever = None

def load_documents_from_directory(directory_path):
    """
    Load all text and PDF files from a directory
    """
    documents = []
    
    # Load text files
    try:
        txt_loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} text files")
    except Exception as e:
        print(f"No text files found or error loading: {e}")
    
    # Load PDF files
    try:
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF files")
    except Exception as e:
        print(f"No PDF files found or error loading: {e}")
    
    return documents

def create_vector_store(documents):
    """
    Create a vector store from documents
    """
    global vectorstore, retriever
    
    if not documents:
        print("No documents to process!")
        return None
    
    print(f"\nProcessing {len(documents)} documents...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
    )
    
    print("✅ Vector store created successfully!\n")
    return vectorstore

def load_sample_data():
    """
    Load sample stock market data/knowledge
    """
    sample_text = """
    Stock Exchange Basics:
    
    A stock exchange is a marketplace where securities, such as stocks and bonds, are bought and sold. 
    The primary function of a stock exchange is to facilitate the buying and selling of securities between investors.
    
    Major Stock Exchanges:
    
    1. New York Stock Exchange (NYSE): The largest stock exchange in the world by market capitalization. 
       Founded in 1792, it's located on Wall Street in New York City.
    
    2. NASDAQ: The second-largest stock exchange globally, known for listing many technology companies. 
       It was the world's first electronic stock market.
    
    3. London Stock Exchange (LSE): One of the oldest stock exchanges, founded in 1801.
    
    4. Tokyo Stock Exchange (TSE): The largest stock exchange in Asia.
    
    Market Indices:
    
    - S&P 500: Tracks 500 large-cap U.S. companies, representing about 80% of the U.S. stock market.
    - Dow Jones Industrial Average (DJIA): Tracks 30 large, publicly-owned companies in the United States.
    - NASDAQ Composite: Includes more than 3,000 stocks listed on the NASDAQ exchange.
    
    Investment Strategies:
    
    1. Buy and Hold: Long-term investment strategy where investors buy stocks and hold them for years.
    2. Value Investing: Buying undervalued stocks based on fundamental analysis.
    3. Growth Investing: Investing in companies expected to grow at an above-average rate.
    4. Dividend Investing: Focusing on stocks that pay regular dividends.
    5. Index Fund Investing: Investing in funds that track market indices like S&P 500.
    
    Risk Management:
    
    - Diversification: Spreading investments across different assets to reduce risk.
    - Asset Allocation: Dividing portfolio among different asset categories.
    - Stop-Loss Orders: Automatic orders to sell when price drops to a certain level.
    - Position Sizing: Determining how much to invest in each position.
    
    Key Financial Metrics:
    
    - P/E Ratio (Price-to-Earnings): Stock price divided by earnings per share.
    - Market Cap: Total value of a company's outstanding shares.
    - Dividend Yield: Annual dividend per share divided by stock price.
    - EPS (Earnings Per Share): Company's profit divided by outstanding shares.
    - ROE (Return on Equity): Net income divided by shareholder equity.
    
    Trading Terms:
    
    - Bull Market: Period of rising stock prices.
    - Bear Market: Period of falling stock prices (typically 20% or more decline).
    - Volatility: Measure of price fluctuations.
    - Liquidity: Ease of buying/selling without affecting price.
    - Market Order: Order to buy/sell immediately at current market price.
    - Limit Order: Order to buy/sell at a specific price or better.
    """
    
    # Create a temporary text file
    temp_dir = "temp_knowledge"
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(f"{temp_dir}/stock_basics.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print("Sample stock market knowledge loaded!\n")
    return temp_dir

def ask_question(question):
    """
    Ask a question using RAG
    """
    try:
        if retriever is None:
            return "Error: Knowledge base not loaded. Please load documents first."
        
        # Get relevant documents
        relevant_docs = retriever.invoke(question)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create messages with chat history
        messages = []
        for msg in chat_history.messages:
            messages.append(msg)
        
        # Add system message with context
        system_msg = system_prompt.format(context=context)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            *[(msg.type, msg.content) for msg in messages],
            ("human", question)
        ])
        
        # Get response
        chain = prompt | llm
        response = chain.invoke({})
        
        # Add to chat history
        chat_history.add_message(HumanMessage(content=question))
        chat_history.add_message(AIMessage(content=response.content))
        
        return response.content
    
    except Exception as e:
        return f"Error: {str(e)}"

def show_conversation_history():
    """
    Display the conversation history
    """
    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)
    
    if not chat_history.messages:
        print("No conversation history yet.")
    else:
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                print(f"\n🧑 You: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"🤖 Chatbot: {msg.content}")
    
    print("=" * 60 + "\n")

def clear_memory():
    """
    Clear the conversation memory
    """
    chat_history.clear()
    print("\n✅ Conversation memory cleared!\n")

def show_knowledge_base_info():
    """
    Show information about the loaded knowledge base
    """
    if vectorstore:
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE INFO")
        print("=" * 60)
        print(f"Vector store: FAISS")
        print(f"Embedding model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"Documents loaded: Yes")
        print("=" * 60 + "\n")
    else:
        print("\n❌ No knowledge base loaded yet.\n")

def main():
    """
    Main chatbot loop
    """
    print("=" * 60)
    print("RAG-Based Stock Exchange Q&A Chatbot")
    print("Powered by LangChain + Groq (LLaMA 3.3 70B) + RAG")
    print("=" * 60)
    
    # Setup knowledge base
    print("\n📚 Setting up knowledge base...")
    print("\nOptions:")
    print("1. Load sample stock market knowledge (quick start)")
    print("2. Load documents from a directory")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Load sample data
        temp_dir = load_sample_data()
        documents = load_documents_from_directory(temp_dir)
        create_vector_store(documents)
    elif choice == "2":
        # Load from user directory
        dir_path = input("Enter directory path containing documents: ").strip()
        if os.path.exists(dir_path):
            documents = load_documents_from_directory(dir_path)
            if documents:
                create_vector_store(documents)
            else:
                print("No documents found. Loading sample data instead...")
                temp_dir = load_sample_data()
                documents = load_documents_from_directory(temp_dir)
                create_vector_store(documents)
        else:
            print("Directory not found. Loading sample data instead...")
            temp_dir = load_sample_data()
            documents = load_documents_from_directory(temp_dir)
            create_vector_store(documents)
    else:
        print("Invalid choice. Loading sample data...")
        temp_dir = load_sample_data()
        documents = load_documents_from_directory(temp_dir)
        create_vector_store(documents)
    
    print("\n🤖 RAG Chatbot ready!")
    print("\nAsk me anything about stock exchanges!")
    print("The chatbot will use both the knowledge base and LLM knowledge.")
    print("\nSpecial commands:")
    print("  - 'history' : Show conversation history")
    print("  - 'clear'   : Clear conversation memory")
    print("  - 'info'    : Show knowledge base info")
    print("  - 'quit'    : Exit the chatbot")
    print("-" * 60 + "\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for special commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using the RAG Stock Exchange Chatbot. Happy investing!")
            break
        
        if user_input.lower() == 'history':
            show_conversation_history()
            continue
        
        if user_input.lower() == 'clear':
            clear_memory()
            continue
        
        if user_input.lower() == 'info':
            show_knowledge_base_info()
            continue
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Get and display response
        print("\nChatbot: ", end="")
        response = ask_question(user_input)
        print(response)
        print()

if __name__ == "__main__":
    main()