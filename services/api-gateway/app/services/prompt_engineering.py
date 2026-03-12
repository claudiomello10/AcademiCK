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

- Before generating the search queries, write a <resolved_query> that restates the student's question with all references resolved (e.g., "that", "it", "the previous topic") using the conversation history. If the query is already self-contained, repeat it as-is.
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


def get_curation_evaluation_prompt(
    query: str,
    subject: str,
    context_chunks: List[Dict],
    iteration: int,
    max_iterations: int,
    previous_reasoning: List[str],
    available_books: List[str]
) -> str:
    """
    Generate the evaluation prompt for the context curation agent.

    The agent evaluates retrieved chunks and decides which to keep/drop,
    optionally requesting additional searches. It never generates the answer.

    The query should already have references resolved (e.g., "explain that further"
    becomes "explain backpropagation further") by the query enhancement step.
    """
    numbered_context = format_context_numbered(context_chunks)

    previous_reasoning_section = ""
    if previous_reasoning:
        previous_reasoning_section = "Previous reasoning:\n" + "\n".join(previous_reasoning) + "\n"

    force_answer_note = ""
    if iteration == max_iterations:
        force_answer_note = "\nIMPORTANT: This is the final iteration. You MUST choose APPROVE.\nKeep the best chunks available — the main LLM will do its best with them.\n"

    books_list = "\n".join(f"- {book}" for book in available_books) if available_books else "No specific books available"

    return f"""You are a context curation agent for an academic RAG system about {subject}.
You do NOT answer the student. Your only job is to decide which retrieved
chunks are relevant and whether more information needs to be fetched.
A separate LLM will generate the final answer using the chunks you approve.

The student asked: "{query}"
Iteration: {iteration}/{max_iterations}

Here are the retrieved context chunks:

{numbered_context}

{previous_reasoning_section}
Your task: Evaluate the retrieved chunks and decide whether they are
sufficient for another LLM to answer the student's query well.
Most retrievals include noise — your job is to curate.

Consider:
- Which chunks are directly relevant to the query?
- Which chunks are noise or off-topic and should be dropped?
- Is any critical information missing that a follow-up search could find?

Respond with ONE action in XML format:

<action type="REFINE">
<reasoning>Brief explanation of what's wrong with current context</reasoning>
<keep_chunks>[comma-separated indices of chunks to KEEP, e.g. 1,3,5]</keep_chunks>
<new_queries>
<query book="all">search query if needed</query>
</new_queries>
</action>

<action type="APPROVE">
<reasoning>Brief explanation of why context is sufficient</reasoning>
<keep_chunks>[comma-separated indices of chunks to KEEP, e.g. 1,2,3,4]</keep_chunks>
</action>

Rules:
- In BOTH actions, you MUST specify which chunks to keep. Unlisted chunks are dropped.
- REFINE must include at least one <query> in <new_queries> to fetch additional context.
- For <query> tags, set book="all" to search all books, or book="exact_book_name" to target a specific book.
- The book name must be written exactly as listed below. Do not omit or add any part of the name.
- Keep only chunks that are directly relevant. Less noise = better final answer.
- If no chunks are relevant, keep an empty list — the system will tell the student the topic was not found in the books.
- Do NOT generate an answer. Only curate the context.

Available books:
{books_list}
{force_answer_note}"""