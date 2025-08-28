
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

query = "What is the main conclusion of the document?"
retrieved_docs = retriever.get_relevant_documents(query)

print(retrieved_docs[0].page_content)

