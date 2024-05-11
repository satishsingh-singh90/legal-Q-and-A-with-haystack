# -*- coding: utf-8 -*-
import streamlit as st
import logging
import time
import os
from haystack.nodes import PDFToTextConverter
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
file_path = "/content/document 2.pdf"
"""logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)"""
#Use this file to set up your Haystack pipeline and querying

#@st.cache_resource(show_spinner=False)

@st.cache_resource(show_spinner=True)     
def preprocess_and_store_documents(file_path):

  if not os.path.exists(file_path):
    raise FileNotFoundError(f"PDF file not found: {file_path}")

  # Preprocess the PDF document
  pdf_converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
  preprocessed = pdf_converter.convert(file_path=file_path, meta={"company": "Company_1", "processed": False})

  # Configure logging (optional)
  logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
  logging.getLogger("haystack").setLevel(logging.INFO)

  # Get Elasticsearch host (optional)
  host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

  # Create and connect to Elasticsearch document store
  document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="document")

  # Delete any existing documents (optional)
  document_store.delete_all_documents()

  # Write the preprocessed document to the store
  document_store.write_documents(preprocessed)

  # Return the document store object
  return document_store



# cached to make index and models load only at start
#@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=True)
def start_haystack_extractive(_document_store):


    # Instantiate your DensePassageRetriever with your own pre-trained model
    retriever = DensePassageRetriever(
        document_store=_document_store,
        query_embedding_model="satishsingh90/deepak_dpr",
        passage_embedding_model="satishsingh90/deepak_dpr"
    )

    reader = FARMReader(model_name_or_path="satishsingh90/deepak_deberta_v3_base", use_gpu=True)
    _document_store.update_embeddings(retriever)

    pipeline = ExtractiveQAPipeline(reader, retriever)

    return pipeline