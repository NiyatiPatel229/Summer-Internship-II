
import os
from langchain_community.embeddings import WatsonxEmbeddings

os.environ["WATSONX_APIKEY"] = "your_api_key"
os.environ["WATSONX_URL"] = "https://abc.com"
os.environ["WATSONX_INSTANCE_ID"] = "abc"

# Load the embeddings model from Watsonx
embeddings = WatsonxEmbeddings(
    model_id="mistralai/mixtral-8x7b-instruct-v01",  
    version="YYYY-MM-DD" 
)

doc_embeddings = embeddings.embed_documents([doc.page_content for doc in split_docs])

