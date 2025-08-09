# import os
# import sys

# # Set working directory
# project_root = "/media/pc1/Ubuntu/Extend_Data/em_Thanh/medrag_colpali"
# os.chdir(project_root)

# # Add project root to sys.path for imports to work
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # Set environment variables
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # if applicable
# os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# import streamlit as st
# import uuid
# import time
# import json
# from PIL import Image
# from src.rag.vectordb.middleware import Middleware

# SESSION_FILE = "data/sessions.json"
# TEMP_IMAGE_DIR = "data/temp_images"
# os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# @st.cache_resource
# def load_middleware():
#     st.write("🔄  Initialising Middleware (should print only once)")
#     return Middleware(
#         kg_path="data/processed/knowledge graph of DDXPlus.xlsx",
#         model_name="vidore/colqwen2-v1.0", 
#         quantized=True, 
#         create_collection=True,
#         enable_rag=True,
#         quantize_llm=False,
#         quantization_type_llm="4bit"
#     )

# middleware = load_middleware()

# # ----------------------------
# # 🔧 Utility Functions
# # ----------------------------
# def save_sessions_to_file():
#     with open(SESSION_FILE, "w", encoding="utf-8") as f:
#         json.dump(st.session_state.sessions, f, indent=2)

# def load_sessions_from_file():
#     if os.path.exists(SESSION_FILE):
#         with open(SESSION_FILE, "r", encoding="utf-8") as f:
#             try:
#                 data = json.load(f)
#                 if not isinstance(data, dict):
#                     return {}
#                 for sess in data.values():
#                     sess.setdefault("image_path", None)
#                     sess.setdefault("query", None)
#                     sess.setdefault("vectordb_docs", None)
#                     sess.setdefault("decomposed", None)
#                     sess.setdefault("kg_results", None)
#                     sess.setdefault("diagnosis", None)
#                     sess.setdefault("messages", [])
#                 return data
#             except Exception:
#                 return {}
#     return {}

# def create_new_session(name: str = None):
#     new_sid = str(uuid.uuid4())
#     session_name = name or f"Session {len(st.session_state.sessions) + 1}"
#     st.session_state.sessions[new_sid] = {
#         "name": session_name,
#         "query": None,
#         "image_path": None,
#         "vectordb_docs": None,
#         "decomposed": None,
#         "kg_results": None,
#         "diagnosis": None,
#         "messages": []
#     }
#     st.session_state.current_session_id = new_sid
#     return new_sid

# def delete_session(session_id):
#     if session_id in st.session_state.sessions:
#         if st.session_state.sessions[session_id].get("image_path"):
#             try:
#                 os.remove(st.session_state.sessions[session_id]["image_path"])
#             except:
#                 pass
#         del st.session_state.sessions[session_id]
#         if st.session_state.current_session_id == session_id:
#             if st.session_state.sessions:
#                 st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
#             else:
#                 create_new_session("Session 1")
#         save_sessions_to_file()

# def save_uploaded_image(image_file):
#     image_id = str(uuid.uuid4())
#     file_path = os.path.join(TEMP_IMAGE_DIR, f"{image_id}.jpg")
#     image = Image.open(image_file)
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image.save(file_path, format="JPEG")
#     return file_path

# def run_backend_pipeline(session):
#     query = session["query"]
#     file_path = session["image_path"]

#     with st.spinner("🔍 ColPali Embedding & Vector DB Search..."):
#         session["vectordb_docs"] = middleware._search_vectordb(query=query, top_k=5)
#         st.success("Documents retrieved")

#     with st.spinner("📡 Feature Decomposition + Knowledge Graph Retrieval..."):
#         results, histories, symptoms = middleware._search_knowledge_graph(query=query, top_k=1)
#         session["decomposed"] = {
#             "history": histories,
#             "symptoms": symptoms
#         }
#         session["kg_results"] = results
#         st.success("Knowledge graph results ready")

#     with st.spinner("🧾 Diagnosis Generation..."):
#         if session["vectordb_docs"] and session["kg_results"]:
#             pdf_paths = [pdf_path['original_file'] for pdf_path in session["vectordb_docs"]] 
#             session["diagnosis"] = middleware.get_answer_from_medgemma(query=query, images_path=[session["image_path"]], retrieved_docs=pdf_paths, kg_info=session["kg_results"])
#             st.success("Diagnosis generated")

#     session["messages"].append({
#         "role": "assistant",
#         "content": f"**🩺 Suggested Diagnosis:**\n{session['diagnosis']}"
#     })
#     save_sessions_to_file()

# # ----------------------------
# # 🔄 App Initialization
# # ----------------------------
# st.set_page_config(page_title="MedRAG: Diagnosis Assistant", layout="wide")

# if "sessions" not in st.session_state:
#     st.session_state.sessions = load_sessions_from_file()

# if "current_session_id" not in st.session_state:
#     if st.session_state.sessions:
#         st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
#     else:
#         create_new_session("Session 1")
#         save_sessions_to_file()

# # ----------------------------
# # 🧾 Sidebar Controls
# # ----------------------------
# st.sidebar.title("📂 Saved Sessions")
# for sid, sess in list(st.session_state.sessions.items()):
#     cols = st.sidebar.columns([4, 1])
#     if cols[0].button(sess["name"], key=sid):
#         st.session_state.current_session_id = sid
#     if cols[1].button("❌", key="delete_" + sid):
#         delete_session(sid)
#         st.rerun()

# if st.sidebar.button("➕ New Session"):
#     create_new_session()
#     save_sessions_to_file()

# # ----------------------------
# # 💬 Main Interface
# # ----------------------------
# session = st.session_state.sessions[st.session_state.current_session_id]

# st.title("🩺 MedRAG: Multimodal Diagnosis Assistant")
# st.caption(f"Current Session: **{session['name']}**")

# # Display message history like ChatGPT
# with st.container():
#     for msg in session["messages"]:
#         with st.chat_message(msg["role"]):
#             if msg["role"] == "image":
#                 st.image(msg["content"], caption="Uploaded Image")
#             else:
#                 st.markdown(msg["content"])


# # Upload image (optional, doesn't trigger backend)
# image_file = st.file_uploader("Upload clinical image (optional)", type=["png", "jpg", "jpeg"])
# if image_file:
#     image_path = save_uploaded_image(image_file)
#     session["image_path"] = image_path
#     session["messages"].append({"role": "user", "content": "📎 Uploaded Image"})
#     session["messages"].append({"role": "image", "content": image_path})

# # Input query and trigger backend
# query = st.chat_input("Enter a clinical case description...")
# if query:
#     session["query"] = query
#     session["messages"].append({"role": "user", "content": query})
#     run_backend_pipeline(session)
#     session["image_path"] = None

# # Display extra diagnosis details if available
# if session.get("diagnosis"):
#     if session.get("vectordb_docs"):
#             with st.expander("Show ColPali Retrieval Results", expanded=False):
#                 for i, doc in enumerate(session["vectordb_docs"]):
#                     st.markdown(f"**Doc {i+1}:** {doc}")
#     else:
#         st.info("No VectorDB documents retrieved yet.")

#     # if session.get("decomposed"):
#     #     with st.expander("Show Feature Decomposition", expanded=False):
#     #         st.json(session["decomposed"])
#     # else:
#     #     st.info("No decomposition results yet.")

#     if session.get("kg_results"):
#         with st.expander("Show Knowledge Graph Results", expanded=False):
#             st.write(session["kg_results"])
#     else:
#         st.info("No knowledge graph results yet.")

# st.markdown("---")
# st.caption("Built with Streamlit | Saved in `sessions.json`")


import os
import sys
import json
import uuid
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional

import gradio as gr
from PIL import Image

# -----------------------------------------------------------------------------
# 🛠  Project-level setup
# -----------------------------------------------------------------------------
project_root = Path("/media/pc1/Ubuntu/Extend_Data/em_Thanh/medrag_colpali")
os.chdir(project_root)

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Hugging Face caches (put *all* caches/temp on your big drive if needed)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # adjust if multi-GPU
os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_TEMP_DIR"] = "/media/pc1/Ubuntu/tmp"

from src.rag.vectordb.middleware import Middleware  # noqa: E402

# -----------------------------------------------------------------------------
# 🗃️  Session storage helpers
# -----------------------------------------------------------------------------
SESS_FILE = project_root / "data/sessions.json"
TEMP_IMG_DIR = project_root / "data/temp_images"
TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

def _load_sessions_from_disk() -> dict:
    if SESS_FILE.exists():
        try:
            data = json.loads(SESS_FILE.read_text("utf-8"))
            if isinstance(data, dict):
                for sess in data.values():
                    sess.setdefault("image_path", None)
                    sess.setdefault("query", None)
                    sess.setdefault("vectordb_docs", None)
                    sess.setdefault("decomposed", None)
                    sess.setdefault("kg_results", None)
                    sess.setdefault("diagnosis", None)
                    sess.setdefault("messages", [])
                return data
        except Exception:
            pass
    return {}

def _save_sessions_to_disk(sessions: dict):
    SESS_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))

# -----------------------------------------------------------------------------
# 🚀 Heavy resources (loaded once per process)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_middleware():
    """Load the Middleware exactly once across all requests."""
    return Middleware(
        kg_path="data/processed/knowledge graph of DDXPlus.xlsx",
        model_name="vidore/colqwen2-v1.0",
        quantized=True,
        create_collection=True,
        enable_rag=True,
        quantize_llm=False,
        quantization_type_llm="4bit",
    )

MW = load_middleware()

# -----------------------------------------------------------------------------
# 🔧 Utility functions
# -----------------------------------------------------------------------------
def _save_uploaded_image(image: Image.Image) -> str:
    img_id = uuid.uuid4().hex
    path = TEMP_IMG_DIR / f"{img_id}.jpg"
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, "JPEG")
    return str(path)

def _messages_for_chatbot(msgs: List[Dict]) -> List[Dict]:
    """
    Chatbot(type='messages') expects OpenAI-style dicts.
    Keep only 'user' and 'assistant' messages.
    """
    return [m for m in msgs if m.get("role") in ("user", "assistant")]

# -----------------------------------------------------------------------------
# 🤖 Core backend pipeline
# -----------------------------------------------------------------------------
def run_pipeline(state: dict, query: str, image_path: Optional[str]):
    state["query"] = query
    state["image_path"] = image_path

    vectordb_docs = MW._search_vectordb(query=query, top_k=5)
    state["vectordb_docs"] = vectordb_docs

    results, histories, symptoms = MW._search_knowledge_graph(query=query, top_k=1)
    state["decomposed"] = {"history": histories, "symptoms": symptoms}
    state["kg_results"] = results

    if vectordb_docs and results:
        pdf_paths = [d["original_file"] for d in vectordb_docs]
        diag = MW.get_answer_from_medgemma(
            query=query,
            images_path=[image_path] if image_path else [],
            retrieved_docs=pdf_paths,
            kg_info=results,
        )
        state["diagnosis"] = diag
        state["messages"].append(
            {"role": "assistant", "content": f"**🩺 Suggested Diagnosis:**\n{diag}"}
        )
    return state

# -----------------------------------------------------------------------------
# 🖼️  UI Components & callbacks
# -----------------------------------------------------------------------------
def ui_build():
    sessions_state = _load_sessions_from_disk()
    if not sessions_state:
        sid = uuid.uuid4().hex
        sessions_state[sid] = {"name": "Session 1", "messages": []}
        _save_sessions_to_disk(sessions_state)

    with gr.Blocks(title="MedRAG Diagnosis Assistant") as demo:
        # States must be inside Blocks
        gr_sessions = gr.State(sessions_state)
        gr_current = gr.State(next(iter(sessions_state)))

        # Header
        gr.Markdown("## 🩺 **MedRAG: Multimodal Diagnosis Assistant**")

        # Session controls
        with gr.Row():
            with gr.Column(scale=3):
                session_names = [v["name"] for v in sessions_state.values()]
                sessions_radio = gr.Radio(
                    choices=session_names,
                    value=session_names[0],
                    label="Saved Sessions",
                )
            with gr.Column(scale=1):
                new_btn = gr.Button("➕ New")
                del_btn = gr.Button("❌ Delete")

        # Chat + inputs
        chatbot = gr.Chatbot(type="messages", show_copy_button=True, height=450)
        image_in = gr.Image(type="pil", label="Clinical image (optional)")
        txt_in = gr.Textbox(placeholder="Enter a clinical case description…", show_label=False)
        submit = gr.Button("Send", variant="primary")

        # Collapsible sections (accordions)
        with gr.Accordion("📄 Show ColPali Retrieval Results", open=False) as acc_vectordb:
            vectordb_out = gr.Textbox(lines=12, interactive=False, label="Retrieved Documents")

        with gr.Accordion("🧠 Show Knowledge Graph Results", open=False) as acc_kg:
            kg_out = gr.Textbox(lines=12, interactive=False, label="Knowledge Graph Info")

        # ---- Helper to refresh Radio after new/delete ----
        def _refresh_session_view(sessions: dict, current_id: str):
            names = [v["name"] for v in sessions.values()]
            current_name = sessions[current_id]["name"] if current_id in sessions else names[0]
            return gr.update(choices=names, value=current_name)

        # ---- Callbacks ----
        def on_new(sessions: dict):
            sid = uuid.uuid4().hex
            sessions[sid] = {"name": f"Session {len(sessions)+1}", "messages": []}
            _save_sessions_to_disk(sessions)
            # clear textboxes + close accordions
            return (
                sessions,
                sid,
                _refresh_session_view(sessions, sid),
                [],
                gr.update(value=""),
                gr.update(value=""),
                gr.update(open=False),
                gr.update(open=False),
            )

        def on_delete(sessions: dict, cur_id: str):
            if cur_id in sessions:
                sessions.pop(cur_id)
                _save_sessions_to_disk(sessions)
            if not sessions:
                sid = uuid.uuid4().hex
                sessions[sid] = {"name": "Session 1", "messages": []}
                cur_id = sid
            else:
                cur_id = next(iter(sessions))
            chat_hist = _messages_for_chatbot(sessions[cur_id]["messages"])
            # clear textboxes + close accordions
            return (
                sessions,
                cur_id,
                _refresh_session_view(sessions, cur_id),
                chat_hist,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(open=False),
                gr.update(open=False),
            )

        def on_select(name: str, sessions: dict):
            sid = next((k for k, v in sessions.items() if v["name"] == name), None)
            if sid is None:
                sid = next(iter(sessions))
            chat_hist = _messages_for_chatbot(sessions[sid]["messages"])
            # close accordions; leave text as-is (or clear if you prefer)
            return (
                sid,
                chat_hist,
                gr.update(open=False),
                gr.update(open=False),
            )

        def on_send(sessions: dict, cur_id: str, query: str, img: Optional[Image.Image]):
            session = sessions[cur_id]
            session["messages"].append({"role": "user", "content": query})

            img_path = _save_uploaded_image(img) if img else None
            if img_path:
                session["image_path"] = img_path
                session["messages"].append({"role": "image", "content": img_path})

            run_pipeline(session, query, img_path)
            _save_sessions_to_disk(sessions)

            chat_hist = _messages_for_chatbot(session["messages"])

            # Fill textboxes
            vdb_text = json.dumps(session.get("vectordb_docs", []), indent=2, ensure_ascii=False) if session.get("vectordb_docs") else ""
            kg_text = (
                json.dumps(session.get("kg_results", ""), indent=2, ensure_ascii=False)
                if isinstance(session.get("kg_results"), (dict, list))
                else (str(session.get("kg_results", "")) if session.get("kg_results") else "")
            )

            # Open accordions iff we have content
            return (
                sessions,
                chat_hist,
                "",  # clear input
                gr.update(value=vdb_text),
                gr.update(value=kg_text),
                gr.update(open=bool(vdb_text)),
                gr.update(open=bool(kg_text)),
            )

        # ---- Wire callbacks (note the extra outputs for accordions) ----
        new_btn.click(
            on_new,
            inputs=[gr_sessions],
            outputs=[gr_sessions, gr_current, sessions_radio, chatbot, vectordb_out, kg_out, acc_vectordb, acc_kg],
        )

        del_btn.click(
            on_delete,
            inputs=[gr_sessions, gr_current],
            outputs=[gr_sessions, gr_current, sessions_radio, chatbot, vectordb_out, kg_out, acc_vectordb, acc_kg],
        )

        sessions_radio.change(
            on_select,
            inputs=[sessions_radio, gr_sessions],
            outputs=[gr_current, chatbot, acc_vectordb, acc_kg],
        )

        submit.click(
            on_send,
            inputs=[gr_sessions, gr_current, txt_in, image_in],
            outputs=[gr_sessions, chatbot, txt_in, vectordb_out, kg_out, acc_vectordb, acc_kg],
        )

    return demo

def main():
    demo = ui_build()
    demo.launch()  # set share=True if you want a public link

if __name__ == "__main__":
    main()

