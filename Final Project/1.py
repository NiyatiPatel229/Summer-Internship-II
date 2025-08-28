
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

# Load a PDF file
pdf_loader = PyPDFLoader("your_file.pdf")
pdf_documents = pdf_loader.load()

# Load a TXT file (if needed)
txt_loader = TextLoader("your_file.txt")
txt_documents = txt_loader.load()

# Load a CSV file (if needed)
csv_loader = CSVLoader(file_path="your_file.csv")
csv_documents = csv_loader.load()

