import gradio as gr
from langchain_ibm import WatsonxLLM  # Or from langchain_community, depending on your setup
from langchain.chains import ConversationalRetrievalChain

llm = WatsonxLLM(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    temperature=0,
    system="Your instructions/system prompt here"
)

# QA Chain: Connect retriever and LLM
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

def chat_fn(query, history):
    # history: [(user, bot), ...]
    response = qa_chain({"question": query, "chat_history": history or []})
    answer = response['answer']
    history = history or []
    history.append((query, answer))
    return answer, history

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Document QA Bot with watsonx Mixtral")
    file_upload = gr.File(label="Upload PDF, TXT, or CSV", file_types=['.pdf', '.txt', '.csv'])
    chatbot = gr.ChatInterface(chat_fn)
    file_upload.change(lambda f: setup_pipeline_with_new_file(f.name), inputs=file_upload)
    chatbot.render()

demo.launch()

