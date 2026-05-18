"""Prompt engineering for RAG responses."""

from typing import List, Dict, Optional


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


def get_enhanced_query_prompt(query: str, subject: str, available_books: List[str], conversation_history: List[Dict] = None) -> str:
    """
    Generate prompt for query enhancement.

    Used to generate multiple focused search queries from a user query.
    """
    books_list = "\n".join(f"- {book}" for book in available_books[:10]) if available_books else "No specific books available"

    conversation_context = ""
    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 6 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                conversation_context += f"<Assistant message>\n{content}\n</Assistant message>\n"
            else:
                conversation_context += f"<User message>\n{content}\n</User message>\n"

    return f"""You are a specialized RAG (Retrieval-Augmented Generation) search term generator. Your task is to generate up to 3 focused search queries between <retrievalX> tags that:

- Target specific textbook content
- Use formal academic terminology
- Focus on fundamental concepts, definitions, theorems
- Break complex queries into core components
- Maximize relevant context retrieval
- Only focus on a specific book if the user requires it
- If a specific book is mentioned in a past message, if its not necessary to use the book, use book="all" or another book.

Guidelines for search queries:

Your queries will be embedded and matched via semantic similarity against textbook chunks stored in a vector database. To get good matches, follow these rules strictly:
- Before generating the search queries, write a <resolved_query> that restates the student's question with all references resolved (e.g., "that", "it", "the previous topic") using the conversation history. The resolved query must be a minimal rewrite — only replace pronouns and references with the actual terms they refer to. Do NOT add information, elaborate, explain concepts, translate, or expand the query beyond what the student wrote. If the query is already self-contained, repeat it exactly as-is. When a <Book>name</Book> tag appears, simply replace it with "the book name" (or "o livro name" if the student is writing in Portuguese). Examples:
  - "Explain <Book>biscect-kmeans</Book>" → "Explain the book biscect-kmeans"
  - "Explique <Book>biscect-kmeans</Book>" → "Explique o livro biscect-kmeans"
  - "What was that concept about?" (previous topic was gradient descent) → "What was gradient descent about?"
- Write queries as declarative statements that resemble how the content would actually be written in a textbook paragraph. DO NOT write commands, instructions, or questions — write statements. The database contains textbook text, so the closer your query looks like actual textbook prose, the better the match.
  - BAD: "Describe the architecture of the multi-task learning model" (this is a command, not textbook text)
  - BAD: "Explain how attention mechanisms weight features for each task" (this is an instruction)
  - GOOD: "The multi-task learning architecture uses a shared encoder with task-specific attention modules" (this resembles textbook prose)
  - GOOD: "Attention mechanisms selectively weight shared features for each task during end-to-end training" (declarative statement)
- Use the specific technical terms, definitions, and formal names that a textbook author would use when explaining the concept.
- DO NOT write structural or navigational queries like "table of contents", "list of chapters", "overview of topic X", "introduction to Y", or "summary of Z" — these will not match any content because the database contains textbook paragraphs, not metadata.
- DO NOT write vague or overly broad queries. Be precise about the specific concept or information the student is asking about.
- Break down complex queries into simpler, core components.
- The search queries should all be focused on the same topic, but each should target a different aspect to maximize coverage.
- It is ok to use similar queries on different retrieval sentences, this will help to find the information in the books.
- The <Book>name</Book> tag in user messages is ONLY a book/article name used to filter which source to search. The text inside the tag is NOT a topic or concept — it is just the title of a book or article. Do not include it in the query text or the resolved query, and do not try to explain it as a concept. For example, "Explain <Book>biscect-kmeans</Book>" means "Explain the contents of the book/article titled 'biscect-kmeans'".
- If a specific book is mentioned in the query using the format <Book>name_of_the_book</Book>, target your search queries to that book by setting book="name_of_the_book".
- The book name should be written exactly as it is written in the tag <Book>name_of_the_book</Book>, do not omit any part of the name, and do not add any part to the name.
- If no specific book is mentioned or if the search should be performed across all available resources, use book="all".
- Focus only on search term generation. Do not provide explanations or answers.
- The subject of the conversation is {subject}.

{conversation_context}

Output format:
<resolved_query>the student's question with all references resolved</resolved_query>
<retrieval1 book="all">search query 1</retrieval1>
<retrieval2 book="book_name">search query 2</retrieval2>
<retrieval3 book="book_name">search query 3</retrieval3>

<Current User Message>
{query}
</Current User Message>

The user response format demands should not affect the search term generation. The search term generation should be focused on generating the search terms that will be used to retrieve the information from the books.
"""


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

    Behavior, rules, and the available-books catalogue. Per-iteration content
    (chunks, iteration counter, previous reasoning) goes in the user prompt.
    """
    books_list = (
        "\n".join(f"- {book}" for book in available_books)
        if available_books
        else "No specific books available"
    )

    return f"""You are a context curation agent for an academic RAG system about {subject}.
You do NOT answer the student. Your only job is to decide which retrieved
chunks are relevant and whether more information needs to be fetched.
A separate LLM will generate the final answer using the chunks you approve.

For each iteration you receive a numbered list of context chunks and must
return a structured decision with these fields:
- action: "APPROVE" when the context is good enough to answer, "REFINE" when
  noise should be dropped and/or more information is needed.
- reasoning: a brief justification.
- keep_indices: 1-indexed chunk numbers to keep. Unlisted chunks are dropped.
  Applies to both APPROVE and REFINE. An empty list is valid when nothing is
  relevant — the system will tell the student the topic was not found.
- new_queries: follow-up search queries. Required (non-empty) for REFINE,
  ignored for APPROVE. Each query has a `query` string and a `book` (exact
  book name from the catalogue below, or null to search all books).

Curation guidelines:
- Keep only chunks that are directly relevant. Less noise = better final answer.
- Most retrievals include irrelevant material — be willing to drop aggressively.
- Consider what critical information is missing that a follow-up search could find.

Query generation rules for REFINE:
Your new queries are embedded and matched by semantic similarity against
textbook chunks in a vector database. To get good matches:
- Write declarative statements that read like textbook prose. Do NOT write
  commands or questions.
  - BAD: "Describe the gradient descent convergence conditions"
  - GOOD: "Gradient descent converges when the learning rate is sufficiently
    small and the loss function is convex"
- Use the specific technical terms a textbook author would use.
- Do NOT write structural/navigational queries like "table of contents",
  "overview of topic X", or "introduction to Y" — these never match content.
- Be precise about what information is missing. Each query should target a
  different aspect to maximize coverage.
- If you know the concept is likely in a specific book, target that book
  instead of searching all. Book names must match the catalogue exactly.

Available books:
{books_list}"""


def get_curation_user_prompt(
    query: str,
    context_chunks: List[Dict],
    iteration: int,
    max_iterations: int,
    previous_reasoning: List[str],
) -> str:
    """Per-iteration user prompt for the curation agent.

    The query should already have references resolved (e.g., "explain that
    further" → "explain backpropagation further") by the query enhancement step.
    """
    numbered_context = format_context_numbered(context_chunks)

    previous_reasoning_section = ""
    if previous_reasoning:
        previous_reasoning_section = (
            "Previous reasoning:\n" + "\n".join(previous_reasoning) + "\n\n"
        )

    force_answer_note = ""
    if iteration == max_iterations:
        force_answer_note = (
            "\nIMPORTANT: This is the final iteration. You MUST choose APPROVE.\n"
            "Keep the best chunks available — the main LLM will do its best with them.\n"
        )

    return f"""The student asked: "{query}"
Iteration: {iteration}/{max_iterations}

Retrieved context chunks:

{numbered_context}

{previous_reasoning_section}Decide whether this context is sufficient for the
main LLM to answer the student's query well. Return your structured decision.
{force_answer_note}"""
