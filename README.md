# Retrieval-Augmented Generation (RAG) for Policy Extraction

This project helps us create and query a searchable knowledge base from plain-text document of the travel directive using OpenAI and LangChain. It’s designed for use cases like summarizing policies or extracting relevant guidance from the document.

---

## Folder Structure

```
project/
├── create_database.py         # Builds the knowledge base from text files
├── query_database.py          # Lets you ask questions about those documents
├── data/                      # Put your .txt documents here
├── chroma/                    # Auto-generated vector database (created by the script)
├── .env                       # Stores your OpenAI API key
├── requirements.txt           # Required packages
└── README.md                  # This file
```

---

##  How to Use the Project

### 1. Install Python

Download from: https://www.python.org/downloads  
During installation, check the box: **"Add Python to PATH"**

---

### 2. Install Dependencies

Open the terminal in VS Code (or any terminal) and run:

```bash
pip install -r requirements.txt
```

---


### 3. Add Your Documents

Documents -> `.txt` files are placed in the `data/` folder.  
Example: `data/tbs-travel-directive-en-2019-11-28.txt` 
For this use case the document that will generate the knowledge base is already there so no need to populate the data directory

---

### 4. Build the Knowledge Base

Run:

```bash
python create_database.py
```

This:
- Loads `.txt` files
- Splits them into chunks
- Embeds them using OpenAI
- Saves to a local vector database (`chroma/`)

---

### 5. Queries

Run:

```bash
python query_database.py
```

Type your query (e.g., _"Alcohol Rules"_).  
The system will:
- Search the most relevant text
- Summarize guidance using OpenAI
- Display sources for transparency

Type `ctrl + c` to quit.

---

### Requirements

All dependencies are listed in `requirements.txt`, including:

- `openai`, `langchain`, `langchain-openai`, `langchain-community`
- `chromadb`, `tiktoken`, `python-dotenv`, `rich`
