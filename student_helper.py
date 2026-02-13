"""Core RAG study assistant class.

Loads ML models, performs hybrid search over embeddings,
builds intent-aware prompts, and generates LLM responses.
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import pipeline as hf_pipeline

from llm_provider import generate_response

# Search weights by intent (dense + sparse)
SEARCH_WEIGHTS = {
    "question_answering": {"dense": 0.6, "sparse": 0.4},
    "summarization": {"dense": 0.7, "sparse": 0.3},
    "coding": {"dense": 0.4, "sparse": 0.6},
    "searching_for_information": {"dense": 0.5, "sparse": 0.5},
}


class StudentHelper:
    """RAG-powered study assistant."""

    def __init__(
        self,
        data_path: str = "data/embeddings.pkl",
        model: str = "gpt-5-mini",
        rag_model: str = "gpt-5-nano",
        subject: str = "Machine Learning",
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = None,
    ):
        """
        Initialize the study assistant.

        Args:
            data_path: Path to the pickle file with embeddings DataFrame
            model: LLM model for final responses
            rag_model: LLM model for query enhancement (should be fast/cheap)
            subject: Study subject (e.g., "Machine Learning")
            embedding_model_name: HuggingFace model name for embeddings
            device: 'cpu' or 'cuda' (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = model
        self.rag_model = rag_model
        self.subject = subject
        self.data_path = data_path
        self.messages = []  # Conversation history

        # Load BGE-M3 embedding model
        print("Loading BGE-M3 embedding model...")
        self.embedding_model = BGEM3FlagModel(
            embedding_model_name,
            device=self.device,
            use_fp16=(self.device == "cuda"),
        )
        print("Embedding model loaded.")

        # Load intent classifier
        print("Loading intent classifier...")
        self.intent_classifier = hf_pipeline(
            "text-classification",
            model="claudiomello/AcademiCK-intent-classifier",
            device=0 if self.device == "cuda" else -1,
        )
        print("Intent classifier loaded.")

        # Load embeddings from disk
        self.embedding_df = None
        self.dense_matrix = None
        self._load_embeddings()

    # ── Data Management ──────────────────────────────────────────────────

    def _load_embeddings(self):
        """Load embeddings DataFrame from pickle and build dense matrix."""
        if not os.path.exists(self.data_path):
            print("No embeddings found. Process PDFs first (option 2).")
            self.embedding_df = None
            self.dense_matrix = None
            return

        with open(self.data_path, "rb") as f:
            self.embedding_df = pickle.load(f)

        if self.embedding_df.empty:
            self.dense_matrix = None
            return

        # Stack all dense embeddings into a numpy matrix for fast search
        self.dense_matrix = np.vstack(self.embedding_df["dense_embedding"].tolist())
        print(f"Loaded {len(self.embedding_df)} chunks from {len(self.get_books())} book(s).")

    def save_embeddings(self, df: pd.DataFrame):
        """Save embeddings DataFrame to pickle, merging with existing data."""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        if self.embedding_df is not None and not self.embedding_df.empty:
            self.embedding_df = pd.concat([self.embedding_df, df], ignore_index=True)
        else:
            self.embedding_df = df

        with open(self.data_path, "wb") as f:
            pickle.dump(self.embedding_df, f)

        # Rebuild dense matrix
        self.dense_matrix = np.vstack(self.embedding_df["dense_embedding"].tolist())

    def reload_embeddings(self):
        """Reload embeddings from disk."""
        self._load_embeddings()

    def get_books(self) -> list[str]:
        """Get list of unique book names."""
        if self.embedding_df is None or self.embedding_df.empty:
            return []
        return sorted(self.embedding_df["Book"].unique().tolist())

    # ── Intent Classification ────────────────────────────────────────────

    def classify_intent(self, query: str) -> str:
        """Classify query intent. Returns one of 4 labels."""
        result = self.intent_classifier(query)[0]
        return result["label"]

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 6,
        book_filter: str = None,
        intent: str = "question_answering",
    ) -> list[dict]:
        """
        Two-stage hybrid search:
        1. Dense pre-filter: top 50 candidates by cosine similarity
        2. Rerank with dense + sparse weighted scores

        Returns list of {text, book, chapter, topic, score}
        """
        if self.dense_matrix is None or self.embedding_df is None:
            return []

        # Optional book filtering
        if book_filter:
            mask = self.embedding_df["Book"].str.contains(book_filter, case=False, regex=False)
            indices = mask[mask].index.tolist()
        else:
            indices = list(range(len(self.embedding_df)))

        if not indices:
            indices = list(range(len(self.embedding_df)))

        # Stage 1: Dense similarity (fast numpy dot product)
        query_enc = self.embedding_model.encode(
            [query], return_dense=True, return_sparse=True
        )
        query_dense = query_enc["dense_vecs"][0]
        query_sparse = query_enc["lexical_weights"][0]

        subset_matrix = self.dense_matrix[indices]
        dense_scores = np.dot(subset_matrix, query_dense)

        # Pre-filter to top 50 candidates
        pre_k = min(50, len(indices))
        top_local = np.argsort(dense_scores)[-pre_k:][::-1]
        candidate_indices = [indices[i] for i in top_local]
        candidate_dense = dense_scores[top_local]

        # Stage 2: Compute sparse scores for candidates
        candidate_texts = [self.embedding_df.iloc[i]["Text"] for i in candidate_indices]
        candidate_sparse = self.embedding_model.encode(
            candidate_texts, return_dense=False, return_sparse=True
        )["lexical_weights"]

        sparse_scores = np.zeros(len(candidate_indices))
        for i, sparse_emb in enumerate(candidate_sparse):
            common_keys = set(sparse_emb.keys()) & set(query_sparse.keys())
            sparse_scores[i] = sum(sparse_emb[k] * query_sparse[k] for k in common_keys)

        # Normalize both to [0, 1]
        dense_norm = _normalize(candidate_dense)
        sparse_norm = _normalize(sparse_scores)

        # Weighted combination by intent
        weights = SEARCH_WEIGHTS.get(intent, {"dense": 0.5, "sparse": 0.5})
        final_scores = weights["dense"] * dense_norm + weights["sparse"] * sparse_norm

        # Return top_k results
        top_k_local = np.argsort(final_scores)[-top_k:][::-1]

        results = []
        for i in top_k_local:
            row = self.embedding_df.iloc[candidate_indices[i]]
            results.append({
                "text": row["Text"],
                "book": row["Book"],
                "chapter": row["Chapter"],
                "topic": row["Topic"],
                "score": float(final_scores[i]),
            })
        return results

    # ── Query Enhancement ────────────────────────────────────────────────

    def generate_enhanced_queries(self, query: str, conversation: bool = False) -> list[dict]:
        """
        Use LLM to generate up to 3 focused search queries.

        Returns list of {"query": str, "book": str|None}
        """
        system_prompt = f"""You are a specialized RAG (Retrieval-Augmented Generation) search term generator. Your task is to generate up to 3 focused search queries between <retrievalX> tags that:

- Target specific textbook content
- Use formal academic terminology
- Focus on fundamental concepts, definitions, theorems
- Break complex queries into core components
- Maximize relevant context retrieval
- Only focus on a specific book if the user requires it
- If a specific book is mentioned in a past message, if its not necessary to use the book, use book="all" or another book.

Guidelines for search queries:

- Use domain-specific technical vocabulary and terminology
- Include key theorems, laws, or principles by their formal names
- Focus on foundational concepts as they would appear in academic texts
- Target textbook sections and chapter topics using standard academic organization
- Break down complex queries into simpler, core components
- Use keywords that maximize relevant context retrieval
- Try to find exactly what the user is looking for
- The search queries should all be focused on the same topic, but they should be different.
- It is ok to use similar queries on different retrieval sentences, this will help to find the information in the books.
- If a specific book is mentioned in the query using the format <Book>name_of_the_book</Book>, target your search queries to that book by setting book="name_of_the_book".
- If no specific book is mentioned or if the search should be performed across all available resources, use book="all".
- Focus only on search term generation. Do not provide explanations or answers.
- The subject of the conversation is {self.subject}.

Output format:
<retrieval1 book="all">search query 1</retrieval1>
<retrieval2 book="book_name">search query 2</retrieval2>
<retrieval3 book="book_name">search query 3</retrieval3>
"""

        # Build user message with optional conversation context
        user_content = ""
        if conversation and self.messages:
            for msg in self.messages[-6:]:
                if msg["role"] == "assistant":
                    user_content += f"<Assistant message>\n{msg['content']}\n</Assistant message>\n"
                else:
                    user_content += f"<User message>\n{msg['content']}\n</User message>\n"

        user_content += f"<Current User Message>\n{query}\n</Current User Message>"
        user_content += "\n\nThe user response format demands should not affect the search term generation. The search term generation should be focused on generating the search terms that will be used to retrieve the information from the books."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response_text = generate_response(messages, self.rag_model, temperature=0.3)
        except Exception as e:
            print(f"Query enhancement failed: {e}. Using original query.")
            return [{"query": query, "book": None}]

        # Parse retrieval tags
        retrievals = []
        for i in range(1, 4):
            pattern = f'<retrieval{i} book="([^"]+)">(.*?)</retrieval{i}>'
            match = re.search(pattern, response_text)
            if match:
                book = match.group(1)
                retrievals.append({
                    "query": match.group(2),
                    "book": None if book == "all" else book,
                })

        if not retrievals:
            return [{"query": query, "book": None}]

        return retrievals

    # ── Prompt Engineering ───────────────────────────────────────────────

    def _get_system_prompt(self, intent: str, context_chunks: list[dict]) -> str:
        """Build the full system prompt: intent instructions + formatted context."""
        instructions = self._get_intent_instructions(intent)
        context = self._format_context(context_chunks)
        return f"{instructions}\n{context}"

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into numbered retrieval blocks."""
        if not chunks:
            return "No relevant context found in the books."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(
                f"Retrieval {i}: From Book: {chunk['book']} - Chapter {chunk['chapter']} - Section: {chunk['topic']}\n{chunk['text']}"
            )
        return "\n\n".join(formatted)

    def _get_intent_instructions(self, intent: str) -> str:
        """Return the intent-specific instruction text."""
        subject = self.subject

        if intent == "question_answering":
            return rf"""You are an assistant helping a student to study {subject}.
The student asks you a question and you provide an answer and an indication on which sections from books given by the retrieval-augmented generation (RAG) context he can learn more about the topics, give the book name chapters and sections that should help him.
When giving him the name of the book, you should provide the full name of the book.
When giving him the name of the chapter, you should provide the full name of the chapter.
When giving him the name of the section, you should provide the full name of the section.
The citation should include the book, chapter, section and page.
If the context has nothing about the topic, tell the student that you could not find the topic in the books, if the context has the topic, provide the information found, if its not too specific you can elaborate a little bit.
If the question is about a specific topic, cite the chapter and section that defines the topic.
If the student asks you a question that requires mathematical calculations do not provide the numerical answer, provide only the method to solve the problem step by step, and instruct him where to find the solution in the book.
If the student asks you about a specific exercise and the context does not provide the problem, ask him to provide the full problem.
If the student asks you about a specific exercise and the context provides the problem, provide the method to solve the problem step by step, and instruct him how to think about the problem. Do not provide the answer.
Focus on making the student think about the problem and how to solve it.
The priority is to make the student learn and understand the topic, not to provide the answer.
Here is the context for the user query retrieved from the books:

"""

        elif intent == "summarization":
            return rf"""You are an assistant helping a student to study {subject}.
The student asks you to help him summarize a specific topic.
If the context has nothing about the topic, tell the student that you could not find the topic in the books.
If the context has the topic, provide a summary of the topic.
If the context has the topic, but its not too specific you can elaborate a little bit.
You will provide him with a summary of the topic.
The summary should be concise and complete.
The summary should be written in a clear and understandable way.
The summary should highlight the most important concepts, definitions, theorems and laws.
The summary should be written in a way that the student can understand the topic without having to read the whole book.
The summary should not include information that is not present in the context.
Always provide the source of the information, the book name, chapter, section and page.
Here is the context for the user query retrieved from the books:

"""

        elif intent == "coding":
            return rf"""You are an assistant helping a student to study {subject}.
You will now help him code a program.
You will provide him with the code and an indicate from which books, chapters and sections the information was retrieved.
When giving him the name of the book, you should provide the full name of the book.
When giving him the name of the chapter, you should provide the full name of the chapter.
When giving him the name of the section, you should provide the full name of the section.
The citation should include the book, chapter, section and page.
If the context has nothing about the topic, tell the student that you could not find the topic in the books, if the context has the topic, provide the information found, if its not too specific you can elaborate a little bit.
The code should be complete and functional.
The code should include comments explaining the code.
The code should be written in a clear and understandable way.
When the language is not specified, use the language that you think is most appropriate.
If the student asks you to code in a specific language, use that language.
When a code in the books is given use the code from the book.
Cite the book, chapter and section where the code was found.
Here is the context for the user query retrieved from the books:

"""

        elif intent == "searching_for_information":
            return rf"""You are an assistant helping a student to study {subject}.
The student asks you to help him find information on a specific topic.
You will provide him with indications on which books, chapters and sections he can learn more about the topics.
When possible, provide a summary of the topic.
When giving him the name of the book, you should provide the full name of the book.
When giving him the name of the chapter, you should provide the full name of the chapter.
When giving him the name of the section, you should provide the full name of the section.
If the context has nothing about the topic, tell the student that you could not find the topic in the books, if the context has the topic, provide the information found, if its not too specific you can elaborate a little bit.
Always provide the book page number, this page number is the PDF page number.
Here is the context for the user query retrieved from the books:

"""

        else:
            return rf"""You are an assistant helping a student to study {subject}.
The student asks you a question and you provide an answer based on the context from books provided by retrieval-augmented generation (RAG).
When giving citations, provide the full name of the book, chapter, and section.
If the context has nothing about the topic, tell the student that you could not find the topic in the books.
Here is the context for the user query retrieved from the books:

"""

    # ── Response Generation ──────────────────────────────────────────────

    def ask(self, query: str) -> str:
        """
        Full RAG pipeline for a query.

        Steps:
        1. Classify intent
        2. Generate enhanced queries
        3. Search with each query, deduplicate
        4. Build system prompt
        5. Assemble messages with conversation history
        6. Call LLM
        7. Update conversation history
        """
        # Step 1: Intent classification
        intent = self.classify_intent(query)
        top_k = 12 if intent == "searching_for_information" else 6

        # Step 2: Enhanced queries
        has_history = len(self.messages) > 0
        enhanced = self.generate_enhanced_queries(query, conversation=has_history)

        # Step 3: Search and deduplicate
        all_results = []
        seen_texts = set()
        for eq in enhanced:
            results = self.search(
                eq["query"], top_k=top_k, book_filter=eq.get("book"), intent=intent
            )
            for r in results:
                text_key = r["text"][:200]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_results.append(r)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:top_k]

        # Step 4: Build system prompt
        system_prompt = self._get_system_prompt(intent, all_results)

        # Step 5: Assemble messages
        llm_messages = []
        if self.messages:
            llm_messages.extend(self.messages[-6:])  # Last 6 for context
        llm_messages.append({"role": "system", "content": system_prompt})
        llm_messages.append({"role": "user", "content": query})

        # Step 6: LLM call
        response = generate_response(llm_messages, self.model)

        # Step 7: Update conversation history
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()

    def set_subject(self, subject: str):
        """Change the study subject."""
        self.subject = subject

    def set_model(self, model: str):
        """Change the LLM model."""
        self.model = model


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    if len(scores) == 0:
        return scores
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)
