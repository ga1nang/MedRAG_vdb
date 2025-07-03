"""
A minimal Gradio UI that lets a user:
    1. Upload a PDF of tropical/infectious diseases or documents
    2. Extract images, embed them with ColPali
    3. Store the vectors in a embedded Qdrant collection
    4. Handle query and retrieve the top_k document
    5. Feed the retrieved result to a RAG back-end
"""
import gradio as gr

from rag.vectordb.middleware import Middleware
from rag.rag import Rag

rag = Rag()


class PDFSearchApp:
    def __init__(self):
        self.indexed_docs: dict[int, bool] = {}
        self.current_pdf = None

    # ---------------- Upload callback ---------------- #
    # inputs=[state, file]  →  parameters (user_id, file)
    def upload_and_convert(self, user_id: int, file) -> str:
        if file is None:
            return "No file uploaded"

        try:
            mw = Middleware(user_id, create_collection=True)
            pages = mw.index(id=user_id, pdf_path=file.name, max_pages=20)  # fixed max_pages for now
            self.indexed_docs[user_id] = True
            return f"Uploaded and indexed {len(pages)} pages."
        except Exception as exc:  # noqa: BLE001
            return f"Error processing PDF: {exc}"

    # ---------------- Search callback --------------- #
    # inputs=[state, query]  →  parameters (user_id, query)
    def search_documents(self, user_id: int, query: str):
        if not self.indexed_docs.get(user_id):
            return "Please index documents first", "--"
        if not query:
            return "Please enter a search query", "--"

        try:
            mw = Middleware(user_id, create_collection=False)
            hits = mw.search([query])[0]          # [(filepath, distance)]
            best_path, _ = hits[0]
            answer = rag.get_answer_from_gemini(query, [best_path])
            return best_path, answer
        except Exception as exc: 
            return f"Error during search: {exc}", "--"


# ---------------- UI ---------------- #
def create_ui():
    app = PDFSearchApp()

    with gr.Blocks() as demo:
        user_id_state = gr.State(value=1)

        gr.Markdown("# ColPali + Qdrant Demo (quick test)")

        with gr.Tab("Upload PDF"):
            file_input = gr.File(label="Upload PDF")
            status_box = gr.Textbox(interactive=False)

        with gr.Tab("Query"):
            query_box = gr.Textbox(label="Enter query")
            search_btn = gr.Button("Query")
            image_out = gr.Image(label="Top page")
            answer_box = gr.Textbox(interactive=False, label="RAG Response")

        # wiring
        file_input.upload(
            fn=app.upload_and_convert,
            inputs=[user_id_state, file_input],
            outputs=[status_box],
        )

        search_btn.click(
            fn=app.search_documents,
            inputs=[user_id_state, query_box],
            outputs=[image_out, answer_box],
        )

    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch()