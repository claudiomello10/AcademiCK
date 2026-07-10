"""Prompt engineering for RAG responses."""

from typing import List, Dict, Optional

from app.config import settings


def get_rag_system_prompt(
    intent: str,
    subject: str,
    context_chunks: List[Dict]
) -> str:
    """
    Generate the system prompt based on intent and context.

    Args:
        intent: Query intent (question_answering, summarization, coding, searching_for_information)
        subject: Study subject (e.g., "Machine Learning")
        context_chunks: Retrieved context chunks from vector search

    Returns:
        System prompt string
    """
    # Format context
    context = format_context(context_chunks)

    # Get intent-specific instructions
    instructions = get_intent_instructions(intent, subject)

    return f"""{instructions}

{context}
"""


def format_context(chunks: List[Dict]) -> str:
    """Format context chunks for the prompt."""
    if not chunks:
        return "No relevant context found in the books."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        book_name = chunk.get('book_name', 'Unknown')
        chapter_title = chunk.get('chapter_title', 'Unknown')
        topic = chunk.get('topic', '')
        text = chunk.get('text', '')
        page_number = chunk.get('page_number')
        page_str = f" - Page: {page_number}" if page_number else ""

        formatted.append(
            f"Retrieval {i}: From Book: {book_name} - Chapter {chapter_title} - Section: {topic}{page_str}\n{text}"
        )

    return "\n\n".join(formatted)


def get_intent_instructions(intent: str, subject: str) -> str:
    """Get intent-specific instructions for the system prompt."""

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
When writing equations or variables, always use the LaTeX format, with the dollar sign at the beginning and end of the equation or variable, or use double dollar signs for equations that should be displayed in a separate line. Examples: $\omega$, $$\lambda$$, $u_1$.
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
When writing equations or variables, always use the LaTeX format, with the dollar sign at the beginning and end of the equation or variable, or use double dollar signs for equations that should be displayed in a separate line. Example: $\omega$ or $$\omega$$.
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
When writing equations or variables, always use the LaTeX format, with the dollar sign at the beginning and end of the equation or variable, or use double dollar signs for equations that should be displayed in a separate line. Example: $\omega$ or $$\omega$$.
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
When writing equations or variables, always use the LaTeX format, with the dollar sign at the beginning and end of the equation or variable, or use double dollar signs for equations that should be displayed in a separate line. Example: $\omega$ or $$\omega$$.
Here is the context for the user query retrieved from the books:

"""

    else:
        return rf"""You are an assistant helping a student to study {subject}.
The student asks you a question and you provide an answer based on the context from books provided by retrieval-augmented generation (RAG).
When giving citations, provide the full name of the book, chapter, and section.
If the context has nothing about the topic, tell the student that you could not find the topic in the books.
When writing equations or variables, always use the LaTeX format. Examples: $\omega$, $$\lambda$$, $u_1$.
Here is the context for the user query retrieved from the books:

"""


def get_enhancement_system_prompt(subject: str) -> str:
    """Static system prompt for the query resolution agent.

    The conversation history is supplied via Pydantic AI's message_history
    parameter; the user query is the user prompt. Retrieval is handled later by
    the curation agent's tools, so this agent only resolves/summarizes the task.
    """
    return f"""You are a query resolution agent for an academic system about {subject}.

Produce a single field:
- resolved_query: the student's question with all references resolved (e.g.,
  "that", "it", "the previous topic") using the conversation history. It must be
  a minimal rewrite — only replace pronouns and references with the actual terms
  they refer to. Do NOT add information, elaborate, explain concepts, translate,
  or expand. If the query is already self-contained, repeat it exactly as-is.
  When a <Book>name</Book> tag appears, simply replace it with "the book name"
  (or "o livro name" if the student writes in Portuguese). Examples:
    - "Explain <Book>biscect-kmeans</Book>" → "Explain the book biscect-kmeans"
    - "Explique <Book>biscect-kmeans</Book>" → "Explique o livro biscect-kmeans"
    - "What was that concept about?" (previous topic was gradient descent)
      → "What was gradient descent about?\""""


def format_context_numbered(chunks: List[Dict]) -> str:
    """Format context chunks with numbered indices and truncated text for the curation agent."""
    if not chunks:
        return "No chunks available."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        book_name = chunk.get('book_name', 'Unknown')
        chapter_title = chunk.get('chapter_title', 'Unknown')
        topic = chunk.get('topic', '')
        text = chunk.get('text', '')
        page_number = chunk.get('page_number')
        page_str = f" - Page: {page_number}" if page_number else ""

        # Truncate text to ~300 chars for cost efficiency
        preview = text[:300] + "..." if len(text) > 300 else text

        formatted.append(
            f"[{i}] From Book: {book_name} - Chapter {chapter_title} - Section: {topic}{page_str}\n    {preview}"
        )

    return "\n\n".join(formatted)


def get_curation_system_prompt(subject: str, available_books: List[str]) -> str:
    """Static system prompt for the curation agent — set once per request.

    Behavior, the tool catalogue with budgets, and the available-books list.
    The query and initial numbered context go in the user prompt.
    """
    books_list = (
        "\n".join(f"- {book}" for book in available_books)
        if available_books
        else "No specific books available"
    )

    return f"""You are a context curation agent for an academic RAG system about {subject}.
You do NOT answer the student. Your job is to assemble the best possible set of
context chunks for a separate LLM that will write the final answer.

You begin with NO retrieved context — gathering it is your job, done entirely
through tools. The numbered context list (shown in the user message, initially
empty) grows as tools add chunks; every tool result shows the new chunks with
their 1-indexed positions and your remaining action budget.

You have ONE shared budget of {settings.agent_max_actions} actions. EVERY tool
call — search or navigation — spends one action. Nothing is free: each call also
appends to the context and makes every later call more expensive in tokens. So
act deliberately and stop as soon as you can answer.

Curate as you go — this is mandatory. Every tool REQUIRES a keep=[indices]
argument: the 1-indexed positions of the currently-live chunks you want to
retain. Any live chunk you do NOT list is permanently dropped in that same move
(no extra action). Every tool result ends with the current "live chunks: [...]"
list — build your next keep from it. A search often returns many chunks and only
a few matter, so your keep should name just those. On your first call nothing is
gathered yet, so pass keep=[]. Dropped chunks leave the context and stop costing
tokens; indices are stable, so dropping never renumbers anything. (New chunks a
tool just added are always kept for now — you review them on your next call.)

Tools (each costs one action):
- search(queries): semantic search — your main way to fetch content. Batch up to
  {settings.agent_max_queries_per_search} declarative, textbook-style queries into
  a single call (each may target a specific book); one call spends one action no
  matter how many queries, so prefer batching over multiple calls.
- list_chapters(books, include_topics=False): the chapter outline of up to 3
  books. By default returns just chapter titles — keep it that way for a broad
  overview. Only pass include_topics=True when you genuinely need each chapter's
  topics in the same call; it is more verbose.
- list_topics(chapters): the topics inside up to 3 chapters.
- read_chapter(book, chapter, mode): read a chapter's introduction (mode="intro")
  or full text (mode="full").
- expand_context(chunk): pull chunks adjacent to a context chunk when a passage
  looks cut off at a boundary.

Strategy — be good without being wasteful:
- For substantive content questions, search is usually your first and best move;
  batch several queries into one search call. For pure structure questions
  ("what is in book X"), a single list_chapters can answer it.
- For broad "what is in book X" questions, stay broad: call list_chapters and
  KEEP the returned outline chunk — that outline IS the answer. Do NOT read or
  search individual chapters for these questions; only go into a specific chapter
  when the student explicitly asks about that chapter or a concept inside it.
- Watch the action counter in every tool result and stop calling tools once the
  context is sufficient — spare actions are not a reason to keep exploring.

Navigation tools add their result to the context list as a numbered chunk and
tell you its index — those outline/topic chunks are real context, so include
their indices in keep_indices whenever they help answer the question.

search() query rules: write declarative textbook-style statements, not questions
or commands. Use precise technical terms. Avoid navigational phrasing like
"table of contents" or "introduction to X" — those never match content.

Final decision (structured output):
- action: "APPROVE" when the assembled context can answer the query, or
  "NOT_IN_KB" when you are confident the topic is absent from all books and more
  searching would be futile (the system then returns a fixed "not found" message).
  Do not use NOT_IN_KB just because chunks are noisy.
- reasoning: a brief justification.
- keep_indices: 1-indexed positions still in context to keep (already-dropped
  chunks are gone). Keep only what the answer needs — less noise, better answer.

Available books:
{books_list}"""


def get_curation_user_prompt(query: str, context_chunks: List[Dict]) -> str:
    """Initial user prompt for the curation agent.

    The query should already have references resolved (e.g., "explain that
    further" → "explain backpropagation further") by the query resolution step.
    """
    if context_chunks:
        context_section = (
            "Current context chunks:\n\n"
            + format_context_numbered(context_chunks)
        )
    else:
        context_section = (
            "No context has been retrieved yet — use your tools to gather it "
            "(search for content; list_chapters / list_topics to explore)."
        )

    return f"""The student asked: "{query}"

{context_section}

Gather what you need with tools, then return your structured decision with the
chunks worth keeping."""
