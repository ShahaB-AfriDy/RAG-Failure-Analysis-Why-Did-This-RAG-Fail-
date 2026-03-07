# RAG Failure Analysis: “Why Did This RAG Fail?”

**Author:** Shahab Afridi

**Project Duration:** Feb 26, 2026 – Mar 7, 2026

---

## 1. Project Purpose

This project implements a **minimal Retrieval-Augmented Generation (RAG) pipeline** to systematically investigate **why RAG systems fail**. The goal is **not** to optimize performance or create a production-ready system, but to:

* Identify and categorize **failure types** (retrieval, generation, ambiguity)
* Understand the impact of **retrieval design parameters** (chunk size, Top-k, MMR, metadata filtering)
* Explore **structural reasoning limitations** of large language model–based RAG systems

---

## 2. Repository Structure

```text
RAG Failure Analysis/
│
├─ corpus.txt                   # Full text corpus for retrieval
├─ 30_Questions.json            # 30 evaluation questions
├─ Rag_Failure_Analysis_experiments.ipynb  # Main RAG experiments notebook
├─ Validation_Experiments.ipynb # Controlled validation experiments
├─ outputs/                     # Generated results and JSON logs
├─ chroma_db/                   # Chroma vector database
├─ requirements.txt             # Python dependencies
├─ .env                         # API key (not tracked)
├─ README.md                     # Project explanation
├─ venv/                        # Optional virtual environment
```

---

## 3. Minimal RAG Implementation

The pipeline includes:

1. **Document Loading**

   * `TextLoader` loads `corpus.txt`.

2. **Text Splitting**

   * Baseline: `chunk_size=150`, `chunk_overlap=20`.
   * Smaller chunk size (`50`) tested to **intentionally expose failures**.

3. **Vector Store**

   * `Chroma` stores document embeddings using `GoogleGenerativeAIEmbeddings`.

4. **Retriever**

   * Retrieval strategies tested:

     * `search_type="similarity"` (default)
     * `search_type="mmr"` with `k=3–5` and `lambda_mult=0.1`
   * Metadata-based filtering applied in experiments to resolve temporal conflicts.

5. **LLM**

   * `ChatGoogleGenerativeAI` (`gemini-2.5-flash-lite`)
   * `temperature=0.0` for deterministic answers, `0.9` for stochastic outputs

---

## 4. Question Design

* **30 questions** cover multiple failure categories:

  * **Multi-document reasoning** – Combining information across several chunks
  * **Temporal conflicts** – Old vs updated policies
  * **Conditional reasoning** – Enrollment year vs credit rules
  * **Ambiguity** – Vague or underspecified terms
  * **Conceptual confusion** – Similar events or terms (e.g., academic meeting vs research symposium)

* Questions are stored in `30_Questions.json` and used for both **baseline and improved retrieval experiments**.

---

## 5. Running the Notebook

1. Create and activate a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

2. Set your Google API key in `.env`:

```text
GOOGLE_API_KEY=your_api_key_here
```

3. Run the notebooks:

```bash
jupyter notebook Rag_Failure_Analysis_experiments.ipynb
jupyter notebook Validation_Experiments.ipynb
```

The notebooks perform:

* Document chunking
* Vector store creation
* Retrieval with similarity/MMR and metadata filtering
* Generation of responses for all 30 questions
* Logging of retrieved chunks, model output, and evaluation

---

## 6. Output Files

* **`outputs/`** contains JSON results for all experiments:

  * `question` – Input query
  * `retrieved_chunks` – Chunks returned by the RAG retriever
  * `answer` – Model-generated output
  * `evaluation` – Correct, partially correct, or incorrect

* This allows **direct reproducibility** of quantitative analyses and failure categorization.

---

## 7. Experimental Design Notes

* **Chunk Size & Overlap** – Smaller chunks fragment context to expose failures
* **Top-k & Retrieval Type** – Tested multiple values to observe retrieval vs generation failures
* **Metadata-Based Filtering** – Used to resolve temporal conflicts
* **LLM Temperature** – `0.9` for stochastic evaluation, `0.0` for deterministic testing
* **Goal** – Capture **structural weaknesses** of RAG rather than optimize performance

---

## 8. Key Findings

* **Generation Errors Predominate** – Most failures occur when the model interprets retrieved content incorrectly, particularly for **conditional and multi-document reasoning**.
* **Retrieval Failures Are Less Frequent** – Missing or fragmented chunks account for fewer failures but can still induce hybrid errors.
* **Ambiguity and Temporal Conflicts** – Metadata filtering mitigates temporal conflicts, but ambiguity remains challenging.

---

## 9. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies include:** `langchain`, `chromadb`, `langchain-google-genai`, `langchain-community`, `python-dotenv`.

---