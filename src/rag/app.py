import streamlit as st
import uuid
import time
import json
import os
from PIL import Image

os.chdir("/media/pc1/Ubuntu/Extend_Data/em_Thanh/medrag_colpali")
os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SESSION_FILE = "data/sessions.json"
TEMP_IMAGE_DIR = "data/temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# ----------------------------
# 🔧 Utility Functions
# ----------------------------
def save_sessions_to_file():
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.sessions, f, indent=2)

def load_sessions_from_file():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
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
                return {}
    return {}

def create_new_session(name: str = None):
    new_sid = str(uuid.uuid4())
    session_name = name or f"Session {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[new_sid] = {
        "name": session_name,
        "query": None,
        "image_path": None,
        "vectordb_docs": None,
        "decomposed": None,
        "kg_results": None,
        "diagnosis": None,
        "messages": []
    }
    st.session_state.current_session_id = new_sid
    return new_sid

def delete_session(session_id):
    if session_id in st.session_state.sessions:
        if st.session_state.sessions[session_id].get("image_path"):
            try:
                os.remove(st.session_state.sessions[session_id]["image_path"])
            except:
                pass
        del st.session_state.sessions[session_id]
        if st.session_state.current_session_id == session_id:
            if st.session_state.sessions:
                st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
            else:
                create_new_session("Session 1")
        save_sessions_to_file()

def save_uploaded_image(image_file):
    image_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_IMAGE_DIR, f"{image_id}.jpg")
    image = Image.open(image_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(file_path, format="JPEG")
    return file_path

def run_backend_pipeline(session):
    query = session["query"]
    file_path = session["image_path"]

    with st.spinner("🔍 ColPali Embedding & Vector DB Search..."):
        time.sleep(1)
        session["vectordb_docs"] = [f"VectorDB Doc #{i+1}" for i in range(3)]
        st.success("Documents retrieved")

    with st.spinner("🧠 MedGemma Feature Decomposition..."):
        time.sleep(1)
        session["decomposed"] = {
            "history": ["recent malaria infection"],
            "symptoms": ["fever", "nausea", "muscle pain"]
        }
        st.success("Features extracted")

    with st.spinner("📡 MedBERT + Knowledge Graph Retrieval..."):
        time.sleep(1)
        session["kg_results"] = ["Malaria", "Dengue", "Zika"]
        st.success("Knowledge graph results ready")

    with st.spinner("🧾 Diagnosis Generation..."):
        time.sleep(1)
        session["diagnosis"] = "Likely Diagnosis: Dengue Hemorrhagic Fever"
        st.success("Diagnosis generated")

    session["messages"].append({
        "role": "assistant",
        "content": f"**🩺 Suggested Diagnosis:**\n{session['diagnosis']}"
    })
    save_sessions_to_file()

# ----------------------------
# 🔄 App Initialization
# ----------------------------
st.set_page_config(page_title="MedRAG: Diagnosis Assistant", layout="wide")

if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions_from_file()

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
    else:
        create_new_session("Session 1")
        save_sessions_to_file()

# ----------------------------
# 🧾 Sidebar Controls
# ----------------------------
st.sidebar.title("📂 Saved Sessions")
for sid, sess in list(st.session_state.sessions.items()):
    cols = st.sidebar.columns([4, 1])
    if cols[0].button(sess["name"], key=sid):
        st.session_state.current_session_id = sid
    if cols[1].button("❌", key="delete_" + sid):
        delete_session(sid)
        st.rerun()

if st.sidebar.button("➕ New Session"):
    create_new_session()
    save_sessions_to_file()

# ----------------------------
# 💬 Main Interface
# ----------------------------
session = st.session_state.sessions[st.session_state.current_session_id]

st.title("🩺 MedRAG: Multimodal Diagnosis Assistant")
st.caption(f"Current Session: **{session['name']}**")

# Display message history like ChatGPT
with st.container():
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "image":
                st.image(msg["content"], caption="Uploaded Image")
            else:
                st.markdown(msg["content"])


# Upload image (optional, doesn't trigger backend)
image_file = st.file_uploader("Upload clinical image (optional)", type=["png", "jpg", "jpeg"])
if image_file:
    image_path = save_uploaded_image(image_file)
    session["image_path"] = image_path
    session["messages"].append({"role": "user", "content": "📎 Uploaded Image"})
    session["messages"].append({"role": "image", "content": image_path})

# Input query and trigger backend
query = st.chat_input("Enter a clinical case description...")
if query:
    session["query"] = query
    session["messages"].append({"role": "user", "content": query})
    run_backend_pipeline(session)
    session["image_path"] = None

# Display extra diagnosis details if available
if session.get("diagnosis"):
    if session.get("vectordb_docs"):
            with st.expander("Show ColPali Retrieval Results", expanded=False):
                for i, doc in enumerate(session["vectordb_docs"]):
                    st.markdown(f"**Doc {i+1}:** {doc}")
    else:
        st.info("No VectorDB documents retrieved yet.")

    if session.get("decomposed"):
        with st.expander("Show Feature Decomposition", expanded=False):
            st.json(session["decomposed"])
    else:
        st.info("No decomposition results yet.")

    if session.get("kg_results"):
        with st.expander("Show Knowledge Graph Results", expanded=False):
            for i, node in enumerate(session["kg_results"]):
                st.markdown(f"**Node {i+1}:** {node}")
    else:
        st.info("No knowledge graph results yet.")

st.markdown("---")
st.caption("Built with Streamlit | Saved in `sessions.json`")
