import streamlit as st
import os
from openai import OpenAI

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Trading AI Dashboard", layout="wide")

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "MashGPT"])

# ------------------------
# DASHBOARD PAGE
# ------------------------
if page == "Dashboard":
    st.title("📈 Trading Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Portfolio Value", "$10,000")
    with col2:
        st.metric("Daily P&L", "+$250")

    st.write("This is your trading dashboard. You can expand this later.")

# ------------------------
# MASHGPT PAGE
# ------------------------
if page == "MashGPT":
    st.title("🤖 MashGPT")

    # Setup OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    client = None

    if api_key:
        try:
            client = OpenAI(api_key=api_key)
        except:
            client = None

    # Chat input
    prompt = st.chat_input("Ask MashGPT anything...")

    if prompt:
        # User message
        with st.chat_message("user", avatar="🙂"):
            st.write(prompt)

        # AI response
        with st.chat_message("assistant", avatar="🤖"):
            st.write("Thinking...")

            if not client:
                st.error("❌ No API key connected")
            else:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are MashGPT, a smart trading assistant that helps with stocks, trading strategies, and general questions."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    reply = response.choices[0].message.content
                    st.write(reply)

                except Exception as e:
                    st.error(f"Error: {e}")
