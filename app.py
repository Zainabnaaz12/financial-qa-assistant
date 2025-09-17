
# app.py
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import re
from utils import (
    validate_file,
    extract_pdf_content,
    extract_excel_content,
    enrich_processed_content,
    build_document_context
)
from llm_client import chat_with_ollama

# Optional: Plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -------------------------------
# Session State Initialization
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "processed_content" not in st.session_state:
    st.session_state.processed_content = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "topic_history" not in st.session_state:
    st.session_state.topic_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# -------------------------------
# Helpers for Metrics
# -------------------------------
def extract_key_metrics_from_content(content: Dict[str, Any]) -> Dict[str, str]:
    metrics = {"revenue": "N/A", "expenses": "N/A", "net_income": "N/A", "margin": "N/A"}
    
    text = " ".join(content.get("text", [])).lower()
    
    if "revenue" in text:
        match = re.search(r"revenue.*?(\d[\d,.]*\d)", text)
        if match:
            metrics["revenue"] = f"${int(match.group(1).replace(',', '')):,}"
    
    if "expenses" in text:
        match = re.search(r"expenses.*?(\d[\d,.]*\d)", text)
        if match:
            metrics["expenses"] = f"${int(match.group(1).replace(',', '')):,}"
            
    if "net income" in text or "net profit" in text:
        match = re.search(r"net income.*?(\d[\d,.]*\d)", text) or re.search(r"net profit.*?(\d[\d,.]*\d)", text)
        if match:
            metrics["net_income"] = f"${int(match.group(1).replace(',', '')):,}"
            
    if "margin" in text:
        match = re.search(r"margin.*?(\d[\d,.]*%)", text)
        if match:
            metrics["margin"] = match.group(1)

    return metrics

def display_metrics(content: Dict[str, Any]):
    st.subheader("📊 Key Financial Metrics")
    m = extract_key_metrics_from_content(content)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", m["revenue"])
    c2.metric("Expenses", m["expenses"])
    c3.metric("Net Income", m["net_income"])
    c4.metric("Margin", m["margin"])

# -------------------------------
# Charts
# -------------------------------
def create_charts(content: Dict[str, Any]):
    if not PLOTLY_AVAILABLE:
        return None, None
    
    # Placeholder data for demonstration
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"]
    revenue = [1200, 1350, 1500, 1680]
    expenses = [800, 900, 950, 1000]
    profit = [400, 450, 550, 680]

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(x=quarters, y=revenue, mode="lines+markers", name="Revenue", line=dict(color="#3b82f6")))
    fig_rev.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#f1f5f9"))

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=quarters, y=expenses, name="Expenses", marker_color="#ef4444"))
    fig_cmp.add_trace(go.Bar(x=quarters, y=profit, name="Profit", marker_color="#10b981"))
    fig_cmp.update_layout(barmode="group", plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#f1f5f9"))

    return fig_rev, fig_cmp

# -------------------------------
# Executive Summary + Questions
# -------------------------------
def generate_executive_summary(content: Dict[str, Any]) -> str:
    prompt = (
        "You are a professional financial analyst. "
        "Write a concise, 3-sentence executive summary of the provided financial data. "
        "Focus on key metrics like revenue, expenses, and profit/net income. "
        "Summarize the most important findings and trends. Avoid raw data dumps and use clear, natural language."
    )
    ctx = build_document_context(content)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Financial content: {ctx}"}
    ]
    resp = chat_with_ollama(messages)
    return resp.get("answer", "Summary unavailable.")

@st.cache_data(show_spinner=False)
def get_suggested_questions(content: Dict[str, Any]) -> List[str]:
    keywords = content.get("financial_keywords", [])
    qs = []
    if "revenue" in keywords or "income" in keywords or "sales" in keywords:
        qs += ["What was the revenue growth?", "How does this quarter's revenue compare to last year?"]
    if "expenses" in keywords or "cost" in keywords or "operating" in keywords:
        qs += ["What are the main expense categories?", "How have expenses changed over time?"]
    if "profit" in keywords or "net income" in keywords or "margin" in keywords:
        qs += ["What is the current profit margin?", "How did profitability change?"]
    qs += ["Summarize the key financial highlights.", "What are the main financial risks?"]
    return list(set(qs))[:6]

# -------------------------------
# Load External Dark Theme CSS
# -------------------------------
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------
# Welcome Page
# -------------------------------
if st.session_state.page == "welcome":
    st.markdown("<h1 class='main-title'>FinanceAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8;'>Upload financial reports and get smart insights.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    with col2:
        if st.button("🎭 Try Demo", use_container_width=True):
            demo = {
                "type": "demo",
                "text": ["Demo financial report. Revenue:$1,250,000. Expenses:$750,000. Net Income:$500,000. Margin:40%"],
                "financial_keywords": ["revenue", "expenses", "profit", "margin", "net income"],
                "numerical_data": ["$1,250,000", "$750,000", "$500,000", "40%"],
                "metadata": {"filename": "Demo_Financial_Report.pdf", "pages": 1},
                "processing_status": "success"
            }
            st.session_state.processed_content = demo
            st.session_state.file_name = demo["metadata"]["filename"]
            st.session_state.chat_history = [{"role": "assistant", "content": "Demo loaded! Ask questions about the data."}]
            st.session_state.summary = None
            st.session_state.page = "results"
            st.rerun()

# -------------------------------
# Upload Page
# -------------------------------
elif st.session_state.page == "upload":
    st.markdown("<h2 style='text-align:center; color:white;'>📤 Upload Your Financial Report</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf", "xlsx"])
    if uploaded_file:
        valid, msg = validate_file(uploaded_file)
        if not valid:
            st.error(msg)
        else:
            with st.spinner("Processing document..."):
                if uploaded_file.name.endswith(".pdf"):
                    content = extract_pdf_content(uploaded_file)
                else:
                    content = extract_excel_content(uploaded_file)
                st.session_state.processed_content = enrich_processed_content(content, uploaded_file)
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.summary = None
                st.session_state.page = "results"
            st.rerun()
    if st.button("⬅️ Back"):
        st.session_state.page = "welcome"
        st.rerun()

# -------------------------------
# Results Page
# -------------------------------
elif st.session_state.page == "results":
    # Chat History Sidebar
    st.sidebar.title("💬 Chat History")
    for topic in st.session_state.topic_history[-10:]:
        st.sidebar.write("• " + topic)
    
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.topic_history = []
        st.rerun()

    if st.session_state.file_name:
     st.markdown(
        f"<div class='file-name-box'>Report: {st.session_state.file_name}</div>",
        unsafe_allow_html=True
    )

    if st.button("⬅️ New Upload"):
        st.session_state.page = "upload"
        st.rerun()

    content = st.session_state.processed_content
    display_metrics(content)

    # Updated tabs - Chat is now a separate tab
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Content", "📊 Charts", "💡 Insights"])

    with tab1:
        # Chat tab with suggested questions and chat interface
        st.markdown("### 💡 Suggested Questions")
        cols = st.columns(3)
        questions = get_suggested_questions(st.session_state.processed_content)
        for i, q in enumerate(questions):
            with cols[i % 3]:
                if st.button(q, key=f"q_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    ctx = build_document_context(st.session_state.processed_content)
                    messages = [
                        {"role": "system", "content": "You are a financial analyst AI. Answer concisely in plain text, format numbers properly, do not copy raw text."},
                        {"role": "user", "content": f"Financial content: {ctx}\n\nQuestion: {q}"}
                    ]
                    resp = chat_with_ollama(messages)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp["answer"]})
                    st.session_state.topic_history.append(q)
                    st.rerun()
        
        st.divider()
        st.markdown("### 💬 Chat with your document")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"<div class='user-msg'>👤 You: {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-msg'>🤖 Assistant: {msg['content']}</div>", unsafe_allow_html=True)
        
        # Chat input - using a form to ensure it's always visible
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question about the document...", key="chat_input")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Generating response..."):
                    ctx = build_document_context(st.session_state.processed_content)
                    messages = [
                        {"role": "system", "content": "You are a financial analyst AI. Answer concisely in plain text, format numbers properly, do not copy raw text."},
                        {"role": "user", "content": f"Financial content: {ctx}\n\nQuestion: {user_input}"}
                    ]
                    resp = chat_with_ollama(messages)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp["answer"]})
                    st.session_state.topic_history.append(user_input)
                st.rerun()
    
    with tab2:
        st.markdown("### Extracted Content")
        if content.get("type") == "pdf":
            for i, page in enumerate(content.get("text", [])):
                st.markdown(f"**Page {i+1}:** {page[:500]}{'...' if len(page) > 500 else ''}")
        elif content.get("type") == "excel":
            for sheet, df in content.get("sheets", {}).items():
                if isinstance(df, pd.DataFrame):
                    st.dataframe(df.head(10))
                else:
                    st.write(df)

    with tab3:
        if PLOTLY_AVAILABLE:
            fig1, fig2 = create_charts(st.session_state.processed_content)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Please install plotly to see charts. `pip install plotly`")

    with tab4:
        st.markdown("### 📝 Executive Summary")
        if st.session_state.summary is None:
            with st.spinner("Generating summary..."):
                st.session_state.summary = generate_executive_summary(st.session_state.processed_content)
        st.write(st.session_state.summary)