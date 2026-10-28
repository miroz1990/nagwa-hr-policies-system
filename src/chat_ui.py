import re
import streamlit as st
from main import ask
from llm import generate_answer

st.set_page_config(page_title="Nagwa HR Policies Q/A Chat", layout="wide")


WELCOME_MESSAGE = (
    "Hi! Ask anything about Nagwa policies. "
    "You can ask about leaves, attendance, maternity, "
    "internal transfer, part-time working, and HR rules."
    "eg: 'How many paid sick leave days are allowed per year?'"
)

def detect_primary_topic(query):
    q = normalize_query(query)

    priority_topics = [
        ("internet", ["internet", "personal usage", "personal use"]),
        ("smoking", ["smoking", "smoke"]),
        ("mobile", ["mobile", "phone"]),
        ("music", ["music", "headphone", "headphones"]),
        ("prayer", ["prayer", "praying", "pray"]),
        ("working hours", ["working hours", "core hours", "flexible working"]),
    ]

    for topic, keywords in priority_topics:
        if any(keyword in q for keyword in keywords):
            return topic

    return None


def normalize_query(query):
    query = query.lower()
    query = query.replace("nagwa", "")
    query = query.replace("in the company", "")
    query = query.replace("at the company", "")
    query = query.replace("company policy", "")
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def tokenize(text):
    return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))


def is_ambiguous_short_question(query):
    q = query.strip().lower()

    vague_phrases = {
        "why", "why?",
        "how", "how?",
        "where", "where?",
        "when", "when?",
        "what", "what?",
        "what about", "what about?",
        "and", "and?",
        "then", "then?",
        "explain",
        "more",
        "details",
        "more details",
    }

    if q in vague_phrases:
        return True

    return q in vague_phrases


def is_follow_up_question(query):
    q = query.strip().lower()

    explicit_followups = {
        "why", "why?",
        "why not?", "why is that?",
        "explain", "explain more",
        "give more details", "more details",
        "details", "what about it?",
        "and?", "then?",
    }

    if q in explicit_followups:
        return True

    followup_starters = (
        "what about",
        "how about",
        "and ",
        "also",
        "then",
        "what if",
        "does that mean",
        "is there",
        "are there",
        "where exactly",
        "when exactly",
        "during",
    )

    if q.startswith(followup_starters):
        return True

    # Only very vague one-word/two-word questions are follow-ups
    vague_short_questions = {
        "why", "why?",
        "where", "where?",
        "when", "when?",
        "how", "how?",
        "what", "what?",
    }

    if q in vague_short_questions:
        return True

    return False

def get_best_source(query, contexts):
    if not contexts:
        return {}

    q = normalize_query(query)
    q_tokens = tokenize(q)

    best_context = contexts[0]
    best_score = -1

    # -------------------------
    # PRIORITY SECTION MATCH
    # -------------------------

    primary_topic = detect_primary_topic(query)

    if primary_topic:
        for ctx in contexts:
            section = ctx.get("section", "").lower()

            if primary_topic == "internet" and "internet" in section:
                return ctx

            if primary_topic == "smoking" and "smoking" in section:
                return ctx

            if primary_topic == "mobile" and "mobile" in section:
                return ctx

            if primary_topic == "music" and (
                "music" in section or "headphone" in section
            ):
                return ctx

            if primary_topic == "prayer" and "prayer" in section:
                return ctx

            if primary_topic == "working hours" and (
                "working hours" in section or "flexible working hours" in section
            ):
                return ctx

    # -------------------------
    # EXISTING SCORING LOGIC
    # -------------------------

    topic_map = {
        "smoking": ["smoking"],
        "internet": ["internet"],
        "mobile": ["mobile phones", "mobile"],
        "phone": ["mobile phones", "mobile"],
        "headphone": ["music players & headphones", "headphones"],
        "music": ["music players & headphones", "music"],
        "prayer": ["prayer times", "prayer"],
        "overtime": ["overtime"],
        "annual leave": ["annual leave", "annual leaves"],
        "sick leave": ["sick leave"],
        "maternity": ["maternity leave"],
        "paternity": ["paternity leave"],
        "part time": ["part-time", "part time"],
        "part-timer": ["part-time", "part time"],
        "insurance": ["medical", "health insurance"],
        "medical": ["medical", "health insurance"],
        "working hours": ["working hours", "flexible working hours"],
    }

    for ctx in contexts:
        section = ctx.get("section", "").lower()
        text = ctx.get("text", "").lower()

        score = 0
        section_tokens = tokenize(section)
        text_tokens = tokenize(text)

        score += len(q_tokens & section_tokens) * 5
        score += len(q_tokens & text_tokens) * 2

        for query_term, section_terms in topic_map.items():
            if query_term in q:
                for term in section_terms:
                    if term in section:
                        score += 10

        score += float(ctx.get("final_score", 0))

        if score > best_score:
            best_score = score
            best_context = ctx

    return best_context

def get_answer_sources(query, answer, contexts, max_sources=3):
    if not contexts:
        return []

    primary_topic = detect_primary_topic(query)

    if any(
        word in normalize_query(query)
        for word in ["insurance", "medical", "health insurance"]
    ):

        insurance_sources = []

        for ctx in contexts:
            text = ctx.get("text", "").lower()

            if (
                "health insurance" in text
                or "medical" in text
                or "insurance" in text
            ):
                insurance_sources.append(ctx)

        if insurance_sources:
            return insurance_sources[:max_sources]

    # Prefer the main policy topic from the question
    if primary_topic:
        topic_sections = {
            "internet": ["internet"],
            "smoking": ["smoking"],
            "mobile": ["mobile phones", "mobile"],
            "music": ["music players & headphones", "music", "headphone"],
            "prayer": ["prayer times", "prayer"],
            "working hours": ["working hours", "flexible working hours"],
        }

        matched_sources = []
        seen = set()

        for ctx in contexts:
            document = ctx.get("document", "Unknown")
            section = ctx.get("section", "Unknown")
            section_lower = section.lower()
            key = (document, section)

            if key in seen:
                continue

            if any(term in section_lower for term in topic_sections.get(primary_topic, [])):
                matched_sources.append(ctx)
                seen.add(key)

        if matched_sources:
            return matched_sources[:max_sources]

    # Fallback: only include sections explicitly mentioned in the answer
    answer_lower = answer.lower()
    selected = []
    seen = set()

    for ctx in contexts:
        document = ctx.get("document", "Unknown")
        section = ctx.get("section", "Unknown")
        key = (document, section)

        if key in seen:
            continue

        section_lower = section.lower()

        if re.search(rf"\b{re.escape(section_lower)}\b", answer_lower):
            selected.append(ctx)
            seen.add(key)

    if selected:
        return selected[:max_sources]

    return [contexts[0]]

def is_greeting_or_goodbye(query):
    q = query.strip().lower()
    return q in {"thanks", "thank you", "bye", "goodbye", "ok", "okay"}

def is_greeting(query):
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "hiya",
        "greetings",
    }

    return query.strip().lower() in greetings

def is_repeated_question(query):
    if not st.session_state.last_question:
        return False

    current = normalize_query(query)
    previous = normalize_query(st.session_state.last_question)

    return current == previous


def build_follow_up_query(user_query):
    return (
        f"The employee previously asked: {st.session_state.last_question}\n\n"
        f"The previous answer was: {st.session_state.last_answer}\n\n"
        f"Now the employee asks: {user_query}\n\n"
        f"Answer this as a follow-up using the same policy context. "
        f"If the previous context contains the explanation, include it."
    )


def reset_state():
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.last_question = None
    st.session_state.last_answer = None
    st.session_state.last_contexts = None
    st.session_state.last_source_query = None


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = None

if "last_source_query" not in st.session_state:
    st.session_state.last_source_query = None


with st.sidebar:
    st.title("⚙️ RAG Settings")
    show_debug = st.checkbox("Show debug info", value=False)

    if st.button("🧹 Clear Chat"):
        reset_state()


st.title("📚 Nagwa HR policies Q/A chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_query = st.chat_input("Ask a question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                contexts = []
                source_query = user_query

                if is_greeting_or_goodbye(user_query):
                    answer = "You’re welcome." if "thank" in user_query.lower() else "Goodbye."
                    contexts = []

                elif is_repeated_question(user_query):
                    answer = (
                        "This appears to be the same question you just asked. "
                        "Please ask a different question or a follow-up question "
                        "such as 'Why?', 'Can you explain more?', or 'What is the policy rationale?'."
                    )
                    contexts = []

                elif is_greeting(user_query):
                    if len(st.session_state.messages) <= 2:
                        answer = (
                            "Hi! It seems like you haven't asked a question yet. "
                            "Please go ahead and ask your question, and I'll do my best "
                            "to provide a clear and concise answer based on Nagwa's HR policies."
                        )
                    else:
                        answer = (
                            "Hello again! Feel free to ask another question about "
                            "Nagwa's HR policies."
                        )

                    contexts = []

                elif is_follow_up_question(user_query):
                    if st.session_state.last_contexts and st.session_state.last_question:
                        effective_query = build_follow_up_query(user_query)
                        contexts = st.session_state.last_contexts
                        answer = generate_answer(effective_query, contexts)
                        source_query = st.session_state.last_source_query or st.session_state.last_question
                    else:
                        answer = (
                            "I’m not sure what this refers to. Could you clarify your question? "
                            "For example: 'Why is smoking not allowed?' or 'Where is smoking not allowed?'"
                        )

                elif is_ambiguous_short_question(user_query) and not st.session_state.last_contexts:
                    answer = (
                        "I’m not sure what this refers to. Could you clarify your question? "
                        "For example: 'Why is smoking not allowed?' or 'Where is smoking not allowed?'"
                    )

                else:
                    answer, contexts = ask(user_query)

                    st.session_state.last_question = user_query
                    st.session_state.last_answer = answer
                    st.session_state.last_contexts = contexts
                    st.session_state.last_source_query = user_query

                st.write(answer)

                NO_ANSWER_PHRASES = [
                    "not mentioned in policy",
                    "not mentioned in this policy",
                    "not found in policy",
                    "information not available",
                ]

                show_sources = not any(
                    phrase in answer.lower()
                    for phrase in NO_ANSWER_PHRASES
                )

                if contexts and show_sources:
                    sources = get_answer_sources(source_query, answer, contexts)

                    if len(sources) == 1:
                        source = sources[0]
                        st.caption(
                            f"Source: {source.get('document', 'Unknown')} | "
                            f"Section: {source.get('section', 'Unknown')}"
                        )
                    else:
                        st.caption("Sources:")
                        for source in sources:
                            st.caption(
                                f"• {source.get('document', 'Unknown')} | "
                                f"Section: {source.get('section', 'Unknown')}"
                            )

                if show_debug and contexts:
                    st.subheader("🔎 Retrieved Contexts")
                    for i, ctx in enumerate(contexts):
                        st.markdown(f"**Chunk {i + 1}:**")
                        st.write(f"Document: {ctx.get('document')}")
                        st.write(f"Section: {ctx.get('section')}")
                        st.write(f"Hybrid Score: {ctx.get('hybrid_score')}")
                        st.write(f"Rerank Score: {ctx.get('rerank_score')}")
                        st.write(f"Section Boost: {ctx.get('section_boost')}")
                        st.write(f"Final Score: {ctx.get('final_score')}")
                        st.code(ctx.get("text", ""))

            except Exception as e:
                answer = f"Error: {str(e)}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})