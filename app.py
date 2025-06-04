import streamlit as st
import tempfile
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

# -------------------- App Setup --------------------
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 QueryBuddy: Chat with SQL DB")

# -------------------- Constants --------------------
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
SUPABASE = "USE_SUPABASE"

radio_opt = [
    "Use SQLite 3 Database (Upload .db file)",
    "Connect to your MySQL Database",
    "Connect to Supabase PostgreSQL"
]

# -------------------- Sidebar Inputs --------------------
with st.sidebar:
    selected_opt = st.radio("Choose the DB which you want to chat", options=radio_opt)

    uploaded_file = None
    mysql_host = mysql_user = mysql_password = mysql_db = supabase_uri = None

    if radio_opt.index(selected_opt) == 1:  # MySQL
        db_uri = MYSQL
        mysql_host = st.text_input("MySQL Host")
        mysql_user = st.text_input("MySQL User")
        mysql_password = st.text_input("MySQL Password", type="password")
        mysql_db = st.text_input("MySQL Database")

    elif radio_opt.index(selected_opt) == 2:  # Supabase
        db_uri = SUPABASE
        supabase_uri = st.text_input(
            "Supabase PostgreSQL URI (e.g. postgresql://user:password@host:5432/db)",
            type="password"
        )

    else:  # SQLite
        db_uri = LOCALDB
        uploaded_file = st.file_uploader("Upload your SQLite DB file", type=["db", "sqlite", "sqlite3"])

    api_key = st.text_input("Groq API Key", type="password")

    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = False

    if not st.session_state.chat_ready:
        if st.button("Start Chat"):
            if api_key:
                st.session_state.chat_ready = True
            else:
                st.warning("Please enter your Groq API key to proceed.")

# -------------------- Check if Ready --------------------
if not api_key or not st.session_state.chat_ready:
    st.stop()

# -------------------- LLM Model Setup --------------------
try:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
except Exception as e:
    st.error("Failed to initialize Groq LLM. Check your API key or model name.")
    st.exception(e)
    st.stop()

# -------------------- DB Config Function --------------------
@st.cache_resource(ttl="2h")
def configure_db(
    db_uri,
    uploaded_file=None,
    mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None,
    supabase_uri=None
):
    try:
        if db_uri == LOCALDB:
            if uploaded_file is None:
                st.warning("Please upload a SQLite database file.")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_db_path = tmp_file.name

            creator = lambda: sqlite3.connect(f"file:{temp_db_path}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite://", creator=creator))

        elif db_uri == MYSQL:
            if not all([mysql_host, mysql_user, mysql_password, mysql_db]):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

        elif db_uri == SUPABASE:
            if not supabase_uri:
                st.error("Please provide the full Supabase PostgreSQL URI.")
                st.stop()
            return SQLDatabase(create_engine(supabase_uri))

    except Exception as db_err:
        st.error("Failed to connect to the database.")
        st.exception(db_err)
        st.stop()

# -------------------- Initialize DB --------------------
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host=mysql_host, mysql_user=mysql_user, mysql_password=mysql_password, mysql_db=mysql_db)
elif db_uri == SUPABASE:
    db = configure_db(db_uri, supabase_uri=supabase_uri)
else:
    db = configure_db(db_uri, uploaded_file=uploaded_file)

# -------------------- LangChain Setup --------------------
try:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
except Exception as setup_err:
    st.error("Failed to initialize LangChain agent or toolkit.")
    st.exception(setup_err)
    st.stop()

# -------------------- Chat UI --------------------
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as chat_error:
            st.error("Something went wrong while processing your query.")
            st.exception(chat_error)
