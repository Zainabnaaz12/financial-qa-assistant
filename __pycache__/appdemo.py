import streamlit as st
import pandas as pd

from utils import (
    extract_pdf_content,
    extract_excel_content,
    validate_file,
    build_document_context,
    enrich_processed_content,
)
from llm_client import chat_with_ollama

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Financial Document Q&A Assistant",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# Custom Styling (Dark Theme)
# -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #1a1a1a;
        color: #f5f5f5;
    }
    
    h1, h2, h3, h4, h5, h6, .chat-header, .tab-header {
        color: #f5f5f5;
    }
    
    .stTabs [data-testid="stTab"] button p {
        font-weight: 600;
        font-size: 1rem;
        color: #a0a0a0;
    }
    
    .stTabs [data-testid="stTab"] button p:hover {
        color: #f5f5f5;
    }
    
    .stTabs [data-testid="stTabList"] button[aria-selected="true"] p {
        color: #4fc0d7;
    }

    .stTextInput>div>div>input, .st-ag {
        background-color: #2a2a2a;
        color: #f5f5f5;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
    }

    /* FIX: Ensure the text area is readable */
    .stTextArea>div>div>textarea {
        background-color: #2a2a2a;
        color: #f5f5f5;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
    }

    /* FIX: Target info/success/warning boxes and style them for the dark theme */
    div.st-dg, div.st-at {
        background-color: #2a2a2a !important;
        border: 1px solid #3d3d3d !important;
        color: #f5f5f5 !important;
    }

    .welcome-card {
        text-align: center;
        padding: 4rem 2rem;
        background-color: #2a2a2a;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid #3d3d3d;
        animation: fadeIn 1s ease-in-out;
    }
    
    .welcome-card h2 {
        color: #4fc0d7;
        margin-top: 0;
        font-weight: 700;
        font-size: 3rem;
        text-shadow: 0 0 15px rgba(79, 192, 215, 0.2);
    }
    .welcome-card p {
        color: #a0a0a0;
        margin-top: 10px;
        font-size: 1.2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4fc0d7 0%, #2980b9 100%);
        color: white;
        font-weight: 600;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(79, 192, 215, 0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(79, 192, 215, 0.5);
    }

    .stFileUploader label {
        color: #f5f5f5;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background-color: #2a2a2a;
        border: 2px dashed #4fc0d7;
        border-radius: 12px;
        padding: 4rem;
        transition: border-color 0.3s;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #f5f5f5;
    }

    .st-chat-message-container {
        border-radius: 12px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .st-chat-message-user {
        background-color: #333333 !important;
        border-radius: 16px 16px 4px 16px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .st-chat-message-assistant {
        background-color: #2a2a2a !important;
        border-radius: 16px 16px 16px 4px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .st-chat-message-user p, .st-chat-message-assistant p {
        color: #e0e0e0;
    }
    
    .chat-container {
        background: #2a2a2a;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    .chat-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: #4fc0d7;
        margin-bottom: 1.5rem;
    }

    .tab-header {
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        color: #f5f5f5;
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #333333;
    }
    .stDataFrame > div > div > div > div > div {
        background-color: #333333 !important;
        color: #f5f5f5 !important;
    }
    .stDataFrame > div > div > div > div > div th {
        background-color: #444444 !important;
        color: #f5f5f5 !important;
    }
    
    .stMarkdown p {
        color: #e0e0e0;
    }
    
    /* FIX: Make the sidebar text readable on a white background */
    .st-emotion-cache-1cpx6a7 p {
        color: black !important;
    }
    
    .st-emotion-cache-1cpx6a7 .sidebar-history {
        background-color: #f5f5f5 !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Session state defaults
# -------------------------------
if "processed_content" not in st.session_state:
    st.session_state.processed_content = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# -------------------------------
# Global Sidebar: Persistent Chat History
# -------------------------------
with st.sidebar:
    st.subheader("💬 Chat History")
    if st.session_state.get("chat_history"):
        st.markdown('<div class="sidebar-history">', unsafe_allow_html=True)
        for m in st.session_state.chat_history:
            role = m.get("role", "user")
            text = m.get("content", "")
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear History", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No conversation yet.")

# -------------------------------
# Main
# -------------------------------
def main():
    st.title("💰 Financial Document Q&A Assistant")

    if not st.session_state.show_uploader and not st.session_state.file_processed:
        show_welcome_screen()
    elif st.session_state.show_uploader and not st.session_state.file_processed:
        show_upload_interface()
    elif st.session_state.file_processed:
        show_results_interface()

# -------------------------------
# Welcome screen
# -------------------------------
def show_welcome_screen():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="welcome-card">
                <h2>📊 Welcome</h2>
                <p>
                Upload a financial document (PDF or Excel) and use natural language to ask questions about revenue, expenses, profits, and more.
                </p>
                <br>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.show_uploader = True
            st.rerun()

# -------------------------------
# Upload interface
# -------------------------------
def show_upload_interface():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.show_uploader = False
            st.rerun()

    with col2:
        st.subheader("📁 Upload Your Financial Document")
        uploaded_file = st.file_uploader(
            "Choose a financial document (PDF, .xlsx, .xls)",
            type=["pdf", "xlsx", "xls"],
            help="Supported formats: PDF, Excel (xlsx, xls) | Max size: 10MB",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            is_valid, message = validate_file(uploaded_file)
            if is_valid:
                st.success(f"✅ {message}")
                if st.button("🚀 Process Document", use_container_width=True):
                    process_document(uploaded_file)
            else:
                st.error(f"❌ {message}")

# -------------------------------
# Results interface
# -------------------------------
def show_results_interface():
    st.markdown(f"<h1 style='font-size: 2rem; color: #4fc0d7;'>Processed Document: {st.session_state.last_uploaded_filename}</h1>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("📁 Upload New File", use_container_width=True):
            reset_state(upload=True)
        if st.button("🏠 Start Over", use_container_width=True):
            reset_state(upload=False)

    display_processed_content()

def reset_state(upload=False):
    st.session_state.file_processed = False
    st.session_state.processed_content = None
    st.session_state.chat_history = []
    st.session_state.show_uploader = upload
    st.session_state.last_uploaded_filename = None
    st.rerun()

def process_document(uploaded_file):
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "pdf":
                content = extract_pdf_content(uploaded_file)
            elif ext in ("xlsx", "xls"):
                content = extract_excel_content(uploaded_file)
            else:
                st.error("Unsupported file format")
                return

            content = enrich_processed_content(content, uploaded_file)
            st.session_state.processed_content = content
            st.session_state.file_processed = True
            st.session_state.last_uploaded_filename = uploaded_file.name

        st.success(f"✅ Successfully processed {uploaded_file.name}!")
        st.rerun()
    except Exception as e:
        st.error(f"Error processing document: {e}")

# -------------------------------
# Display content + chat input
# -------------------------------
def display_processed_content():
    content = st.session_state.processed_content
    if not content:
        st.warning("No content extracted.")
        return

    tabs = st.tabs(["💬 Q&A Chat", "📄 Raw Content", "📊 Data Tables", "💰 Insights", "📋 Summary"])

    # --- Q&A Chat tab ---
    with tabs[0]:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-header">💬 Q&A Chat</div>', unsafe_allow_html=True)

        # Show chat history inside the tab too
        for m in st.session_state.chat_history:
            role = m.get("role", "user")
            text = m.get("content", "")
            try:
                st.chat_message(role).markdown(text)
            except Exception:
                st.markdown(f"**{role.capitalize()}:** {text}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Content tab ---
    with tabs[1]:
        st.markdown('<div class="tab-header">📄 Extracted Content (Preview)</div>', unsafe_allow_html=True)
        if content.get("type") == "pdf" and content.get("text"):
            for i, page_text in enumerate(content["text"][:5]):
                st.subheader(f"Page {i+1}")
                st.text_area(f"", page_text, height=300, disabled=True, key=f"page_text_{i}", help="This is a raw text extraction from the PDF page.")
        elif content.get("type") == "excel" and content.get("sheets"):
            names = list(content["sheets"].keys())
            if names:
                selected = st.selectbox("Choose a sheet:", names)
                st.dataframe(content["sheets"][selected].head(20).style.highlight_max(axis=0))
            else:
                st.info("No sheets found in this Excel file.")

    # --- Tables tab ---
    with tabs[2]:
        st.markdown('<div class="tab-header">📊 Extracted Tables / Sheets</div>', unsafe_allow_html=True)
        if content.get("type") == "pdf":
            tables = content.get("tables", [])
            if tables:
                for idx, df in enumerate(tables):
                    st.write(f"Table {idx+1}")
                    st.dataframe(df)
            else:
                st.info("No tables detected in PDF.")
        elif content.get("type") == "excel":
            sheets = content.get("sheets", {})
            if sheets:
                for sheet_name, df in sheets.items():
                    st.write(f"Sheet: {sheet_name}")
                    st.dataframe(df.head(20))
            else:
                st.info("No sheets found in Excel.")

    # --- Insights tab ---
    with tabs[3]:
        st.markdown('<div class="tab-header">💰 Financial Insights</div>', unsafe_allow_html=True)
        kw = content.get("financial_keywords", [])
        nums = content.get("numerical_data", [])
        if kw:
            st.info("**Detected Financial Terms:**")
            st.write(", ".join(kw))
        else:
            st.info("No financial keywords detected.")
        if nums:
            st.info("**Numbers Found (sample):**")
            st.write(", ".join(nums[:50]))
        else:
            st.info("No numerical values detected.")

    # --- Summary tab ---
    with tabs[4]:
        st.markdown('<div class="tab-header">📋 Document Summary</div>', unsafe_allow_html=True)
        st.json(
            {
                "Type": content.get("type", "unknown"),
                "Pages": content.get("pages", None),
                "Sheets": content.get("metadata", {}).get("sheet_count", None),
                "Detected Tables": len(content.get("tables", []) if content.get("tables") else []),
                "Keywords Found": len(content.get("financial_keywords", [])),
            }
        )
    
    # ✅ Chat input OUTSIDE tab container (always active)
    user_input = st.chat_input("Ask about revenue, profit, expenses, etc...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        doc_context = build_document_context(content)
        system_prompt = (
            "You are a financial assistant. Use ONLY the provided document context to answer. "
            "If the answer is not present, respond: 'I don't know'. "
            "Provide concise answers and cite page or sheet if possible.\n\nDOCUMENT CONTEXT:\n"
            + doc_context
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        with st.spinner("Thinking..."):
            try:
                resp = chat_with_ollama(messages, stream=False)
            except Exception as e:
                resp = f"[LLM Error] {e}"

        answer_text = ""
        if isinstance(resp, dict):
            answer_text = resp.get("answer") or resp.get("message", {}).get("content", "")
        else:
            answer_text = str(resp)

        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
        st.rerun()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()








# ------------------------------- # PAGE CONFIGURATION # ------------------------------- st.set_page_config( page_title="FinanceAI - Financial Document Q&A Assistant", page_icon="💰", layout="wide" ) # FIXED CSS with better visibility st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap'); /* Global Dark Theme - Fixed visibility issues */ .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"], section[data-testid="stSidebar"] { background-color: #0a0a0a !important; color: #ffffff !important; font-family: 'Inter', sans-serif !important; } /* App Title - Always visible */ [data-testid="stMarkdownContainer"] h1 { font-size: 3.5rem !important; font-weight: 900 !important; text-align: center !important; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #10b981 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; margin: 2rem 0 !important; display: block !important; } /* Subheadings - Always visible */ [data-testid="stMarkdownContainer"] h2 { font-size: 2.5rem !important; font-weight: 700 !important; color: #ffffff !important; text-align: center !important; margin: 1.5rem 0 !important; } /* Section headings - Always visible */ [data-testid="stMarkdownContainer"] h3 { font-size: 1.5rem !important; font-weight: 600 !important; color: #3b82f6 !important; margin: 1rem 0 !important; } /* All text - Always visible */ [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] div, .stMarkdown div, .stMarkdown p, .stMarkdown span, .element-container p, .element-container div, .element-container span, p, div, span, label, text { color: #e2e8f0 !important; } /* Button text - always visible */ button, button span, button div, button p { color: #ffffff !important; } /* Tooltip text visibility */ [data-testid="stTooltipHoverTarget"], .stTooltip, [role="tooltip"], [data-baseweb="tooltip"] { background: #1a1a1a !important; color: #ffffff !important; border: 1px solid #3b82f6 !important; } /* Force all text elements to be visible */ * { color: inherit !important; } /* Specific overrides for common invisible text */ .stMarkdown, .stMarkdown *, .element-container, .element-container *, [data-testid="stText"], [data-testid="stText"] *, .stText, .stText * { color: #e2e8f0 !important; } /* Force ALL containers to black background */ .stMarkdown, .element-container, [data-testid="element-container"], .stAlert, div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"], .block-container, .main .block-container > div, .stColumn, [data-testid="column"] { background-color: #000000 !important; } /* FORCE all button backgrounds to be black - stronger selectors */ button[kind="secondary"], button[kind="primary"], .stButton > button, [data-testid="stButton"] button, .st-emotion-cache-* button, div[data-testid="stButton"] > button { background: #000000 !important; background-color: #000000 !important; color: #ffffff !important; border: 2px solid #334155 !important; border-radius: 12px !important; } /* Button hover states */ button[kind="secondary"]:hover, button[kind="primary"]:hover, .stButton > button:hover, [data-testid="stButton"] button:hover { background: #000000 !important; background-color: #000000 !important; border-color: #3b82f6 !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important; } /* Fix text area content visibility with stronger selectors */ textarea[disabled], .stTextArea textarea[disabled], [data-testid="stTextArea"] textarea, .st-emotion-cache-* textarea { background: #000000 !important; background-color: #000000 !important; color: #ffffff !important; opacity: 1 !important; } /* Override Streamlit's default disabled text styling */ textarea:disabled { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; opacity: 1 !important; } /* Force all element containers to black */ [class*="st-emotion-cache"] { background: #000000 !important; background-color: #000000 !important; } /* Override any white backgrounds in containers */ .stContainer, .css-1d391kg, .css-18e3th9, .css-1dp5x4i, .css-12oz5g7, .css-1v0mbdj { background-color: #000000 !important; } /* Buttons - Always visible with blue gradient */ [data-testid="stButton"] > button { background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important; color: #ffffff !important; border: none !important; border-radius: 12px !important; padding: 0.875rem 2rem !important; font-weight: 600 !important; box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important; transition: all 0.3s ease !important; } [data-testid="stButton"] > button:hover { transform: translateY(-2px) !important; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.6) !important; border: 2px solid #06b6d4 !important; } /* File Uploader - Black with blue border on hover */ [data-testid="stFileUploader"] { background: #000000 !important; border: 2px solid #334155 !important; border-radius: 16px !important; padding: 3rem 2rem !important; text-align: center !important; transition: all 0.3s ease !important; } [data-testid="stFileUploader"]:hover { border-color: #3b82f6 !important; box-shadow: 0 0 20px rgba(59, 130, 246, 0.3) !important; } [data-testid="stFileUploader"] label { color: #e2e8f0 !important; font-weight: 500 !important; } /* Tabs - Always visible text, blue highlight when active */ .stTabs { background: #000000 !important; border-radius: 16px !important; padding: 1.5rem !important; border: 1px solid #334155 !important; } .stTabs [data-baseweb="tab-list"] { background: #111111 !important; border-radius: 12px !important; padding: 0.5rem !important; } .stTabs [data-baseweb="tab-list"] button { background: transparent !important; color: #e2e8f0 !important; border: none !important; border-radius: 8px !important; padding: 0.75rem 1.5rem !important; font-weight: 500 !important; transition: all 0.3s ease !important; } .stTabs [data-baseweb="tab-list"] button:hover { background: #1a1a1a !important; color: #ffffff !important; border: 1px solid #3b82f6 !important; } .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important; color: #ffffff !important; box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4) !important; border: none !important; } /* Chat Messages - Black with blue borders */ .stChatMessage { background: #000000 !important; border: 1px solid #334155 !important; border-radius: 16px !important; padding: 1.5rem !important; margin: 1rem 0 !important; } /* Chat Input - Always visible */ [data-testid="stChatInputContainer"] { background: #000000 !important; border: 2px solid #334155 !important; border-radius: 16px !important; transition: all 0.3s ease !important; } [data-testid="stChatInputContainer"]:focus-within { border-color: #3b82f6 !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important; } [data-testid="stChatInputContainer"] textarea { background: transparent !important; color: #ffffff !important; border: none !important; font-size: 1rem !important; } [data-testid="stChatInputContainer"] textarea::placeholder { color: #94a3b8 !important; } /* Metrics Cards - Black with blue borders on hover */ [data-testid="metric-container"] { background: #000000 !important; border: 1px solid #334155 !important; border-radius: 12px !important; padding: 1.5rem !important; transition: all 0.3s ease !important; } [data-testid="metric-container"]:hover { border-color: #3b82f6 !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.3) !important; transform: translateY(-2px) !important; } [data-testid="metric-container"] [data-testid="metric-label"] { color: #94a3b8 !important; font-weight: 500 !important; font-size: 0.875rem !important; } [data-testid="metric-container"] [data-testid="metric-value"] { color: #3b82f6 !important; font-size: 2rem !important; font-weight: 700 !important; } /* Data Tables - Always visible */ .stDataFrame { background: #000000 !important; border-radius: 12px !important; border: 1px solid #334155 !important; } .stDataFrame table { background: #000000 !important; color: #ffffff !important; } .stDataFrame thead th { background: #111111 !important; color: #3b82f6 !important; font-weight: 600 !important; border-bottom: 2px solid #3b82f6 !important; } .stDataFrame tbody td { background: #000000 !important; color: #e2e8f0 !important; border-bottom: 1px solid #334155 !important; } .stDataFrame tbody tr:hover { background: #111111 !important; } /* Alerts - Always visible */ .stSuccess { background: rgba(16, 185, 129, 0.1) !important; border-left: 4px solid #10b981 !important; color: #10b981 !important; border-radius: 12px !important; } .stError { background: rgba(239, 68, 68, 0.1) !important; border-left: 4px solid #ef4444 !important; color: #ef4444 !important; border-radius: 12px !important; } .stWarning { background: rgba(245, 158, 11, 0.1) !important; border-left: 4px solid #f59e0b !important; color: #f59e0b !important; border-radius: 12px !important; } .stInfo { background: rgba(59, 130, 246, 0.1) !important; border-left: 4px solid #3b82f6 !important; color: #3b82f6 !important; border-radius: 12px !important; } /* Text Areas - Always visible */ .stTextArea textarea { background: #000000 !important; color: #ffffff !important; border: 1px solid #334155 !important; border-radius: 8px !important; } /* Fix content visibility in text areas */ .stTextArea textarea[disabled] { background: #000000 !important; color: #ffffff !important; opacity: 1 !important; } /* Chat input styling - fix white background */ [data-testid="stChatInput"] { background: #000000 !important; } [data-testid="stChatInput"] > div { background: #000000 !important; border: 2px solid #334155 !important; border-radius: 16px !important; } [data-testid="stChatInput"] input { background: #000000 !important; color: #ffffff !important; border: none !important; } /* Additional chat input container fixes */ .stChatInput { background: #000000 !important; } .stChatInput > div > div { background: #000000 !important; border: 2px solid #334155 !important; } /* Welcome container styling */ .welcome-container { background: #000000 !important; border: 1px solid #334155 !important; border-radius: 1rem !important; padding: 2rem !important; margin-top: 2rem !important; text-align: center !important; } </style> """, unsafe_allow_html=True) # Display plotly status only if not available if not PLOTLY_AVAILABLE: st.sidebar.warning("⚡ Charts disabled. Install plotly: pip install plotly==5.15.0")