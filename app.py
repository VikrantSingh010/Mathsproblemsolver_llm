import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set the app configuration and layout
st.set_page_config(
    page_title="Math & Data Search Assistant",
    page_icon="🧮",
    layout="centered"
)

# Title and description
st.title("📚 Text to Math Problem Solver")
st.markdown("### Your all-in-one math solver and knowledge assistant! 💡")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection dropdown
    model_options = ["Gemma2-9b-It", "llama3-8b-8192", "mixtral-8x7b-32768"]  # Example models
    selected_model = st.selectbox("Select Groq Model", model_options, help="Choose the Groq model to use.")
    
    # API key input
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API Key to access advanced features.")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# Initialize the selected Language Model
llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key)

# Wikipedia tool for data search
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching information on the topics mentioned."
)

# Calculator for math operations
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only input mathematical expressions need to be provided."
)

# Prompt template for reasoning tasks
prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation in points for the question below.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Reasoning tool for logical questions
chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the assistant agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm your Math and Knowledge Assistant! Ask me anything about math or search for information on any topic."}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User question input
st.markdown("### 💬 Ask a question:")
question = st.text_area("Type your math problem or question below:", height=100, placeholder="e.g., I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. How many pieces of fruit do I have at the end?")

# Button for generating an answer
if st.button("Solve"):
    if question.strip():
        with st.spinner("Finding the answer..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            # Callback handler for processing the response
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            # Display the response in the chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.success(response)

    else:
        st.warning("Please enter a question to get started.")

# Footer for extra information
st.markdown("---")
st.markdown("**Math & Data Search Assistant** - Powered by LangChain and Gemma2-9b. 🚀")
