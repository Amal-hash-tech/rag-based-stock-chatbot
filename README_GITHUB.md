\# 🤖 RAG-Based Stock Exchange Chatbot



A powerful Retrieval-Augmented Generation (RAG) chatbot for stock market analysis, powered by LangChain, Groq's LLaMA 3.3 70B, and FAISS vector database.



\[!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

\[!\[LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)

\[!\[License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



\## 🎯 Features



\- 📚 \*\*Document-Aware\*\*: Load and query PDFs, text files about stock markets

\- 🔍 \*\*Semantic Search\*\*: AI-powered similarity search using embeddings

\- 💾 \*\*Vector Storage\*\*: Fast FAISS database for efficient retrieval

\- 🧠 \*\*Conversation Memory\*\*: Maintains context across conversations

\- 🎓 \*\*Sample Data Included\*\*: Built-in stock market knowledge base

\- 🔒 \*\*Privacy-First\*\*: All document processing happens locally

\- ⚡ \*\*Powered by LLaMA 3.3 70B\*\*: State-of-the-art language model via Groq



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.8 or higher

\- Groq API key (\[Get it free here](https://console.groq.com/))



\### Installation



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/YOUR\_USERNAME/rag-stock-chatbot.git

cd rag-stock-chatbot

```



2\. \*\*Install dependencies\*\*

```bash

pip install -r requirements\_rag.txt

```



3\. \*\*Set up environment variables\*\*



Create a `.env` file in the project root:

```env

GROQ\_API\_KEY=your\_groq\_api\_key\_here

```



4\. \*\*Run the chatbot\*\*

```bash

python rag\_stock\_chatbot.py

```



\## 📖 Usage



\### Option 1: Quick Start with Sample Data

```bash

python rag\_stock\_chatbot.py

\# Choose option 1 when prompted

```



The chatbot will load built-in stock market knowledge covering:

\- Stock exchange basics

\- Major exchanges (NYSE, NASDAQ, LSE, TSE)

\- Market indices (S\&P 500, DJIA, NASDAQ)

\- Investment strategies

\- Risk management

\- Financial metrics



\### Option 2: Use Your Own Documents

```bash

python rag\_stock\_chatbot.py

\# Choose option 2 when prompted

\# Enter the path to your documents folder

```



Supported file formats:

\- `.txt` - Text files

\- `.pdf` - PDF documents



\### Example Conversation



```

You: What is the S\&P 500?



Chatbot: Based on the knowledge base, the S\&P 500 tracks 500 large-cap 

U.S. companies, representing about 80% of the U.S. stock market...



You: How does it compare to the Dow Jones?



Chatbot: Comparing the S\&P 500 (which I just mentioned) to the Dow Jones 

Industrial Average - the DJIA tracks only 30 large companies...

```



\### Special Commands



While chatting, you can use these commands:

\- `history` - View conversation history

\- `clear` - Clear conversation memory

\- `info` - Show knowledge base information

\- `quit` - Exit the chatbot



\## 🏗️ Architecture



```

┌─────────────┐

│   User      │

│  Question   │

└──────┬──────┘

&nbsp;      │

&nbsp;      ↓

┌─────────────────┐

│   Embeddings    │  (Convert to vectors)

└──────┬──────────┘

&nbsp;      │

&nbsp;      ↓

┌─────────────────┐

│  FAISS Vector   │  (Similarity search)

│    Database     │

└──────┬──────────┘

&nbsp;      │

&nbsp;      ↓

┌─────────────────┐

│   Retriever     │  (Get top 3 relevant docs)

└──────┬──────────┘

&nbsp;      │

&nbsp;      ↓

┌─────────────────┐

│  LLaMA 3.3 70B  │  (Generate answer)

│   via Groq      │

└──────┬──────────┘

&nbsp;      │

&nbsp;      ↓

┌─────────────────┐

│     Answer      │

└─────────────────┘

```



\## 🛠️ Technology Stack



\- \*\*LLM\*\*: LLaMA 3.3 70B (via Groq)

\- \*\*Framework\*\*: LangChain

\- \*\*Embeddings\*\*: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)

\- \*\*Vector Store\*\*: FAISS (Facebook AI Similarity Search)

\- \*\*Document Loaders\*\*: PyPDF, TextLoader



\## 📁 Project Structure



```

rag-stock-chatbot/

├── rag\_stock\_chatbot.py      # Main chatbot application

├── requirements\_rag.txt       # Python dependencies

├── .env                       # Environment variables (create this)

├── .gitignore                 # Git ignore rules

├── README.md                  # This file

├── RAG\_GUIDE.md              # Detailed documentation

└── RAG\_QUICKSTART.txt        # Quick reference guide

```



\## ⚙️ Configuration



\### Customizing Retrieval



Edit `rag\_stock\_chatbot.py` to customize:



\*\*Number of retrieved documents:\*\*

```python

retriever = vectorstore.as\_retriever(

&nbsp;   search\_kwargs={"k": 5}  # Change from 3 to 5

)

```



\*\*Chunk size:\*\*

```python

text\_splitter = RecursiveCharacterTextSplitter(

&nbsp;   chunk\_size=1500,  # Change from 1000

&nbsp;   chunk\_overlap=300  # Change from 200

)

```



\*\*Memory window:\*\*

```python

chat\_history = WindowChatMessageHistory(k=20)  # Keep 20 messages

```



\## 🔒 Security \& Privacy



\- ✅ API keys stored in `.env` file (never committed to Git)

\- ✅ All document processing happens locally

\- ✅ Only questions sent to Groq API (not documents)

\- ✅ Vector embeddings generated on your machine

\- ✅ FAISS database stored locally



\## 🐛 Troubleshooting



\### Common Issues



\*\*Issue\*\*: `ModuleNotFoundError: No module named 'faiss'`

```bash

pip install faiss-cpu

```



\*\*Issue\*\*: `ModuleNotFoundError: No module named 'sentence\_transformers'`

```bash

pip install sentence-transformers

```



\*\*Issue\*\*: Slow first run

\- First run downloads the embedding model (~80MB)

\- Subsequent runs are fast as the model is cached



\*\*Issue\*\*: Out of memory

\- Reduce chunk size in the code

\- Process fewer documents at once



\## 📊 Performance



\- \*\*Embedding Model Size\*\*: ~80MB (downloaded on first run)

\- \*\*Average Query Time\*\*: 1-3 seconds

\- \*\*Memory Usage\*\*: ~500MB-1GB (depends on document size)

\- \*\*Supported Documents\*\*: Unlimited (limited by disk space)



\## 🤝 Contributing



Contributions are welcome! Please feel free to submit a Pull Request.



1\. Fork the repository

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 🙏 Acknowledgments



\- \[LangChain](https://python.langchain.com/) - Framework for LLM applications

\- \[Groq](https://groq.com/) - Ultra-fast LLM inference

\- \[FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search

\- \[HuggingFace](https://huggingface.co/) - Sentence transformers



\## 📧 Contact



Your Name - your.email@example.com



Project Link: \[https://github.com/YOUR\_USERNAME/rag-stock-chatbot](https://github.com/YOUR\_USERNAME/rag-stock-chatbot)



\## 🗺️ Roadmap



\- \[ ] Add support for more document formats (DOCX, CSV)

\- \[ ] Implement persistent vector storage

\- \[ ] Add web scraping for real-time stock data

\- \[ ] Create web interface with Streamlit

\- \[ ] Add multi-language support

\- \[ ] Implement source attribution in responses



---



\*\*⭐ If you find this project useful, please consider giving it a star!\*\*

