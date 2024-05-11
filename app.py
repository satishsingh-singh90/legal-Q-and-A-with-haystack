import streamlit as st
import os
import logging
import subprocess
import tempfile
from haystack_utility import preprocess_and_store_documents, start_haystack_extractive
import pandas as pd

# Function to install Elasticsearch
@st.cache_resource(show_spinner=True)
def install_elasticsearch():
    command = "wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q && \
               tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz && \
               sudo chown -R daemon:daemon elasticsearch-7.9.2 && \
               sudo -u daemon -- elasticsearch-7.9.2/bin/elasticsearch &"

    try:
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        st.error(f"Error installing Elasticsearch: {e}")
        return False

# Streamlit UI
def main():
    st.title("Run Shell Script")

    if st.button("Install Elasticsearch"):
        st.write("Installing Elasticsearch...")
        if install_elasticsearch():
            st.success("Elasticsearch installed successfully!")
        else:
            st.error("Failed to install Elasticsearch.")

    st.title("Question Answering from PDF")
    file = st.file_uploader("Upload PDF", type="pdf")
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    if file is not None:
        temp_file_path = os.path.join("/content", "uploaded_pdf.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(file.read())

        try:
            document_store = preprocess_and_store_documents(temp_file_path)
            st.write("Document store created and populated successfully!")
        except FileNotFoundError as e:
            st.write(f"Error: {e}")
        pipe=start_haystack_extractive(document_store)

        question1 = "Highlight the parts (if any) of this contract related to \"Document Name\" that should be reviewed by a lawyer. Details: The name of the contract"

        question2 = "Highlight the parts (if any) of this contract related to \"Parties\" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract"

        question3 = "Highlight the parts (if any) of this contract related to \"Agreement Date\" that should be reviewed by a lawyer. Details: The date of the contract"

        question4 = "Highlight the parts (if any) of this contract related to \"Effective Date\" that should be reviewed by a lawyer. Details: The date when the contract is effective"

        question5 = "Highlight the parts (if any) of this contract related to \"Expiration Date\" that should be reviewed by a lawyer. Details: On what date will the contract's initial term expire?"

        st.write("Question 1")
        if st.button(question1):
            st.write("Question 1")
            get_answer(pipe, question1)
        st.write("Question2")
        if st.button(question2):
            st.write("question2")
            get_answer(pipe, question2)
        st.write("Question3")
        if st.button(question3):
            get_answer(pipe, question3)
        st.write("Question4")
        if st.button(question4):
            get_answer(pipe, question4)
        st.write("Question5")
        if st.button(question5):
            get_answer(pipe, question5)

def get_answer(pipe, question):
    answers = []
    prediction = pipe.run(query=question,
                          params={"Retriever": {"top_k": 10},
                                  "Reader": {"top_k": 1}})
    answers.append(prediction)
    for answer in answers:
        st.write("Q:", answer["query"])
        if answer["answers"]:
            st.write("A:", answer["answers"][0].answer)
            st.write("\n")
        else:
            st.write("No answers found")


if __name__ == "__main__":
    main()
