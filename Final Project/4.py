# vectordb.png

from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(split_docs, embeddings)

