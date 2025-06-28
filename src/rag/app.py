"""
A minimal Gradio UI that lets a user:
    1. Upload a PDF of tropical/infectious diseases or documents
    2. Extract images, embed them with ColPali
    3. Store the vectors in a embedded Qdrant collection
    4. Handle query and retrieve the top_k document
    5. Feed the retrieved result to a RAG back-end
"""
import gradio as gr

from vectordb.middleware import Middleware
from rag.rag import Rag

rag = Rag()

class PDFSearchApp:
    def __init__(self):
        self.indexed_docs = {}
        self.current_pdf = None
        
    def upload_and_convert(self, state: int, file):
        if file is None:
            return "No file uploaded"
        
        try:
            self.current_pdf = file.name
            middleware = Middleware(id, create_collection=True)
            pages = middleware.index(id=state, pdf_path=file.name)
            self.indexed_docs[state] = True
            
            return f"Uploaded and extracted {len(pages)} pages" 
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def search_document(self, query: str, state: int = 1, num_result: int = 1):
        print(f"Searching for query: {query}")
        
        if not self.indexed_docs[id]:
            print("Please index documents first")
            return "Please index documents first", "--"
        if not query:
            print("Please enter a search query")
            return "Please enter a search query", "--"
        
        try:
            middleware = Middleware(state, create_collection=False)
            search_results = middleware.search([query])[0]
            page_num = search_results[0][1] + 1
            
            print(f"Retrieved page number: {page_num}")
            img_path = f"pages/{id}/page_{page_num}.png"
            print(f"Retrieved image path: {img_path}")
            rag_response = rag.get_answer_from_gemini(query, [img_path])
            return rag_response
        except Exception as e:
            return f"Error during search: {str(e)}", "--"
        
def create_ui():
    app = PDFSearchApp()
    
    with gr.Blocks() as demo:
        state = gr.State(value={"user_uuid": None})

        gr.Markdown("# Colpali Qdrant Multimodal RAG Demo")

        with gr.Tab("Upload PDF"):
            with gr.Column():
                file_input = gr.File(label="Upload PDF")
                
                max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=10,
                    label="Max pages to extract and index"
                )
                
                status = gr.Textbox(label="Indexing Status", interactive=False)
        
        with gr.Tab("Query"):
            with gr.Column():
                query_input = gr.Textbox(label="Enter query")
                search_btn = gr.Button("Query")
                llm_answer = gr.Textbox(label="RAG Response", interactive=False)
                images = gr.Image(label="Top page matching query")
        
        # Event handlers
        file_input.change(
            fn=app.upload_and_convert,
            inputs=[state, file_input, max_pages_input],
            outputs=[status]
        )
        
        search_btn.click(
            fn=app.search_documents,
            inputs=[state, query_input],
            outputs=[images, llm_answer]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()