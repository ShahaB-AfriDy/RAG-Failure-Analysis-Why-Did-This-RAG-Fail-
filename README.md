
---

# RAG Failure Analysis: “Why Did This RAG Fail?”

**Author:** Shahab Afridi

**Project Duration:** Feb 11, 2026 – Feb 21, 2026

---

## 1. Project Purpose

This project implements a **minimal Retrieval-Augmented Generation (RAG) pipeline** to study **why and how RAG systems fail**. The goal is **not** to optimize performance or build a production-ready system, but to:

* Understand **when and why RAG fails**
* Analyze failures in a structured, reproducible way
* Reflect on **retrieval vs generation vs ambiguity limitations**

---

## 2. Repository Structure

```
RAG Failure Analysis/
│
├─ .env                     # API keys (not tracked in Git)
├─ README.md                # Project explanation
├─ corpus.txt               # Text corpus used for retrieval
├─ Questions_List.txt       # 12 designed RAG failure questions
├─ Rag_Failure_Analysis.ipynb  # Minimal RAG implementation notebook
├─ outputs/                 # JSON results from experiments
│    ├─ rag_results_similarity.json
│    └─ rag_results_mmr.json
├─ chroma_db/               # Chroma vector database
└─ venv/                    # Virtual environment (optional)
```

---

## 3. Minimal RAG Implementation

The pipeline includes:

1. **Document Loading**

   * Using `TextLoader` to load `corpus.txt`.
2. **Text Splitting**

   * Initially `chunk_size=150` for baseline, then reduced to `chunk_size=50` with `chunk_overlap=5` to **intentionally expose failures**.
3. **Vector Store**

   * `Chroma` database stores document embeddings using `GoogleGenerativeAIEmbeddings`.
4. **Retriever**

   * Configurable retrieval strategies:

     * `search_type="similarity"` (default)
     * `search_type="mmr"` (optional) with `k=3` and `lambda_mult=0.1`
   * Top-k retrieval tested for `k=3` and `k=5`.
5. **LLM**

   * `ChatGoogleGenerativeAI` (`gemini-2.5-flash-lite`) with `temperature=0.9` to allow **diverse, creative answers**.

---

## 4. Question Design

* 12 questions were manually crafted to explore **different failure modes**:

  * **Multi-document reasoning**: Combining information from multiple documents
  * **Temporal conflicts**: Old vs updated rules
  * **Ambiguity**: Confusing or incomplete references
  * **Conceptual confusion**: Similar concepts with subtle differences
  * **Structural reasoning limitations**

* Questions are listed in `Questions_List.txt`.

---

## 5. Running the Notebook

1. Create a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

2. Set your Google API key in `.env`:

```
GOOGLE_API_KEY=your_api_key_here
```

3. Run the notebook:

```bash
jupyter notebook Rag_Failure_Analysis.ipynb
```

4. The notebook will:

   * Split documents into chunks
   * Build the vector store
   * Run all 12 questions
   * Save results in JSON format under `outputs/`

---

## 6. Output Files

* **`outputs/rag_results_similarity.json`**: Answers using similarity search (top-k=5)
* **`outputs/rag_results_mmr.json`**: Answers using MMR retrieval (k=3, lambda_mult=0.1)
* Each JSON contains:

  * `question`
  * `retrieved_chunks`
  * `answer`

---

## 7. Notes on Experimental Design

* **Chunk Size & Overlap**: Reduced intentionally to fragment critical context and induce retrieval and generation failures.
* **Top-k & Retrieval Type**: Tested different retrieval configurations to observe hybrid failures and structural limitations.
* **LLM Temperature**: 0.9 to generate diverse answers, allowing analysis of creative or unexpected failures.
* **Goal**: Observe **RAG failures clearly**, not to maximize accuracy.

---

## 8. What This Project Does Not Cover

* Performance optimization of retrieval or generation
* Production-ready RAG pipelines
* Extensive fine-tuning or embedding strategies

This is strictly a **research-focused failure analysis**.

---