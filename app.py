"""Run with: streamlit run app.py"""
from pathlib import Path
import hashlib
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4

from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
import streamlit as st

from connection import create_client
from rag import load_chunks, classify, embed, retrieve, generate, fallback

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Customer Email Automation", page_icon="✉️",
                   layout="centered", initial_sidebar_state="collapsed")
st.set_option("client.toolbarMode", "minimal")
# Visual presentation includes both sidebar navigation states.
st.markdown(f"<style>{(ROOT / 'styles.css').read_text()}</style>", unsafe_allow_html=True)
st.title("Customer Email Automation")

try:
    chunks = load_chunks(ROOT / "data")
except (OSError, ValueError):
    st.error("Company policies are unavailable. Please contact the administrator.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    with st.expander("Connection"):
        st.caption("HKUST GenAI")
        api_key = st.text_input("API key", type="password",
                                value=os.getenv("HKUST_API_KEY", ""))
        model = os.getenv("CHAT_MODEL", "gpt-5.6-luna")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    top_k = 4
    min_score = 0.2
    st.session_state.setdefault("history", [])

    def restore_history(record_id):
        record = next(r for r in st.session_state["history"] if r["id"] == record_id)
        st.session_state["customer_name"] = record["name"]
        st.session_state["email_subject"] = record["subject"]
        st.session_state["email_body"] = record["body"]
        st.session_state["result"] = record

    with st.expander("History"):
        if not st.session_state["history"]:
            st.caption("No replies yet.")
        for record in reversed(st.session_state["history"]):
            st.button(record["subject"], key="history_" + record["id"],
                      on_click=restore_history, args=(record["id"],), use_container_width=True)
            st.caption(record["created_at"])

st.subheader("Customer email")
with st.form("customer_email"):
    name = st.text_input("Customer name", key="customer_name", max_chars=45,
                         placeholder="Customer name")
    subject = st.text_input("Subject", key="email_subject", max_chars=100,
                            placeholder="Email subject")
    body = st.text_area("Email body", key="email_body", height=220, max_chars=5000,
                        placeholder="Paste the customer's email here")
    submitted = st.form_submit_button("Generate reply", type="primary")

if submitted:
    st.session_state.pop("result", None)
    if not all(x.strip() for x in (name, subject, body)):
        st.error("Please enter the customer name, subject and email body.")
    elif not api_key.strip() or not model.strip() or not embedding_model.strip():
        st.error("Please complete the connection in Settings (API key).")
    else:
        try:
            client = create_client(api_key)
            with st.spinner("Writing your reply…"):
                category = classify(client, model.strip(), subject, body)
                hits = []
                if category != "other":
                    # Session-local cache: no raw keys or emails on disk/global cache.
                    signature = hashlib.sha256((api_key + embedding_model + repr(chunks)).encode()).hexdigest()
                    if st.session_state.get("index_signature") != signature:
                        vectors = embed(client, embedding_model.strip(), [c.text for c in chunks])
                        st.session_state["vectors"] = vectors
                        st.session_state["index_signature"] = signature
                    query = embed(client, embedding_model.strip(), [subject + "\n" + body])[0]
                    hits = retrieve(chunks, st.session_state["vectors"], query, category, top_k, min_score)
                reply = generate(client, model.strip(), name.strip(), subject, body, hits) if hits else fallback(name.strip())
                record = {"id": uuid4().hex, "category": category, "hits": hits,
                          "reply": reply, "name": name.strip(), "subject": subject,
                          "body": body, "model": model, "embedding_model": embedding_model,
                          "created_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M")}
                st.session_state["history"].append(record)
                st.session_state["result"] = record
            st.rerun()
        except AuthenticationError:
            st.error("Unable to connect. Please check the API key in Settings.")
        except RateLimitError:
            st.error("The service is busy or your account quota has been reached. Please try again later.")
        except APIConnectionError:
            st.error("Unable to connect. Please check your connection and try again.")
        except APIStatusError:
            st.error("The service is unavailable. Please check the connection settings or try again later.")
        except ValueError:
            st.error("We couldn’t create a reply. Please try again.")

if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()
    st.subheader("Reply")
    if not result["hits"]:
        st.info("This request needs a support team review.")
    st.text_area("Reply draft", result["reply"], height=300, disabled=True,
                 label_visibility="collapsed")
    st.download_button("Download reply", result["reply"], "reply.txt", "text/plain")
