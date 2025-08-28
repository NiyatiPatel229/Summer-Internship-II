from langchain.text_splitter import RecursiveCharacterTextSplitter

documents = pdf_documents

# Create and apply the text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

print(split_docs[0].page_content)

