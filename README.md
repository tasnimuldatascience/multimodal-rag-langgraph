# Multimodal Retrieval-Augmented Generation (RAG) using LangGraph

This repository provides a robust and modular implementation of a Multimodal Retrieval-Augmented Generation (RAG) system leveraging **LangGraph**, **ChromaDB**, **Streamlit**, and **CLIP embeddings** for multimodal embeddings. This project serves as a strong portfolio piece showcasing your expertise in building Generative AI solutions.

---

## 🚀 Overview

The Multimodal RAG application integrates:

- **Multimodal embeddings:** CLIP (Contrastive Language-Image Pretraining) for unified text and image embeddings.
- **Vector Database:** ChromaDB for efficient embedding storage and retrieval.
- **Multi-agent architecture:** Built using LangGraph, orchestrating agents for retrieval and generation.
- **User interface:** Streamlit web application for easy interaction.

---

## 🌟 Features

- **Multimodal Queries:** Accepts both textual and image queries.
- **Retrieval-Augmented Generation:** Provides contextual answers leveraging retrieved knowledge from an embedding database.
- **Modular and Scalable Codebase:** Organized clearly into agents, graphs, embedding models, and utilities.

---

## 📂 Project Structure

```
multimodal-rag-langgraph/
├── agents/
│   ├── retrieval_agent.py
│   └── generation_agent.py
├── data/
│   ├── coco_dataset.py
│   └── wikipedia_dataset.py
├── embeddings/
│   └── multimodal.py
├── graphs/
│   └── agent_graph.py
├── utils/
│   ├── config.py
│   └── vector_store.py
├── .env
├── populate_db.py
├── app.py
└── requirements.txt
```

---

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/tasnimuldatascience/multimodal-rag-langgraph.git
cd multimodal-rag-langgraph
```

### Step 2: Create and activate environment

```bash
conda create -n multimodal-rag python=3.10
conda activate multimodal-rag
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Create a `.env` file:
```bash
OPENAI_API_KEY='your-api-key'
```

---

## 🗃️ Populate the Database

Populate your embedding database using Wikipedia articles and images from the COCO dataset:

```bash
python populate_db.py
```

---

## 🎯 Run the Application

Start your Streamlit application:

```bash
streamlit run app.py
```

Access your application at `http://localhost:8501`

---

## 🧑‍💻 Usage

- Select modality (**text** or **image**).
- Enter a textual query or upload an image along with a question.
- Generate contextually relevant answers.

---

## 🛠️ Technologies Used

- **LangGraph** (Multi-agent orchestration)
- **ChromaDB** (Vector storage)
- **CLIP** (Multimodal embeddings)
- **Streamlit** (Web interface)
- **OpenAI API** (Generation agent)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgements

- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI](https://openai.com/)
- [ChromaDB](https://docs.trychroma.com/)

