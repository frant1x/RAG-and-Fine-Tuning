import streamlit as st
from secret import API_KEY
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = API_KEY


class RAG:
    def __init__(self):
        # self.data_folder = "documents/"
        self.persist_directory = "db/"
        # self.db = self.create_db()
        # print("Saving your db as pickle file...")
        # self.db.persist()
        # print("Saved!")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-3-small",
            chunk_size=64,
            timeout=60,
            show_progress_bar=True,
            retry_min_seconds=15,
        )
        self.db = Chroma(
            persist_directory=self.persist_directory, embedding_function=embeddings
        )

        self.PROMPT_TEMPLATE = """
        Дайте відповідь на запитання, спираючись виключно на наступний контекст:
        {context}
        - -
        Дайте відповідь на запитання на основі вищевказаного
        контексту: {question}
        """
        # user_interface
        st.title("AI ChatBot (RAG)")

        user_input = st.text_input(
            "please enter a message here.",
            key="user_input",
            on_change=self.retriev_data,
        )

    def create_db(self):
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".pdf"):
                print(f"reading {filename}...")
                # Construct the full path to the PDF file
                file_path = os.path.join(self.data_folder, filename)

                # Load the PDF document
                loader = PyPDFLoader(file_path)
                raw_documents = loader.load()

                # Split the text from the document into chunks
                text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
                split_documents = text_splitter.split_documents(raw_documents)

                db = Chroma.from_documents(
                    split_documents,
                    OpenAIEmbeddings(
                        deployment="text-embedding-3-small",
                        chunk_size=3,
                        timeout=60,
                        show_progress_bar=True,
                        retry_min_seconds=15,
                    ),
                    persist_directory=self.persist_directory,
                )
        return db

    def retriev_data(self):
        # retrieving context from db
        user_input = st.session_state["user_input"]
        context_docs = self.db.similarity_search(user_input)
        context = "\n".join([doc.page_content for doc in context_docs[:3]])  # only 3

        prompt = ChatPromptTemplate.from_template(template=self.PROMPT_TEMPLATE)
        messages = prompt.format_messages(question=user_input, context=context)
        response = self.llm.invoke(messages)
        st.write(f"🤖: {response.content}")
        st.session_state["user_input"] = ""


class FineTuning:
    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY)
        # user_interface
        st.title("AI ChatBot (Fine-Tuning)")

        user_input = st.text_input(
            "please enter a message here.",
            key="user_input",
            on_change=self.retriev_data,
        )

    def retriev_data(self):
        response = self.client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:test::AwBrAGn2",
            messages=[
                {
                    "role": "user",
                    "content": st.session_state["user_input"],
                }
            ],
        )
        st.write(f"🤖: {response.choices[0].message.content}")
        st.session_state["user_input"] = ""


def main():
    sidebar = st.sidebar

    page = sidebar.radio("Select Chatbot", ["RAG", "Fine-Tuning"])

    if page == "RAG":
        RAG()
    elif page == "Fine-Tuning":
        FineTuning()


if __name__ == "__main__":
    main()
