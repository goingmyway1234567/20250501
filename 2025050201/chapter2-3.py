# chapter2-3.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# .envからAPIキーを読み込む
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlitページ設定
st.set_page_config(
    page_title="My Great ChatGPT",
    page_icon="🤗"
)
st.header("My Great ChatGPT 🤗")

# チャット履歴の初期化
if "message_history" not in st.session_state:
    st.session_state.message_history = [
        ("system", "You are a creative assistant.絶対に関西弁で返答してください")
    ]

# ユーザー入力の受付
user_input = st.chat_input("聞きたいことを入力してね！")

if user_input:
    with st.spinner("ChatGPT is typing ..."):

        # 共通プロンプトテンプレートと出力パーサーを定義
        prompt = ChatPromptTemplate.from_messages([
            *st.session_state.message_history,
            ("user", "{user_input}")
        ])
        output_parser = StrOutputParser()

        # temperatureごとに応答を生成・表示
        for temperature in [0, 0.5, 1]:  # 0ほど一貫性あり，1に近づくほど創造的になる
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",         # ★ 修正: モデル名を明示
                temperature=temperature,
                api_key=api_key                # ★ 修正: openai_api_key → api_key
            )
            chain = prompt | llm | output_parser
            response = chain.invoke({"user_input": user_input})

            st.subheader(f"🌡 temperature = {temperature}")
            st.markdown(response)

            # 最初の応答だけ履歴に追加（必要なら全温度分を保存してもよい）
            if temperature == 0:
                st.session_state.message_history.append(("user", user_input))
                st.session_state.message_history.append(("ai", response))

# チャット履歴の表示（ページ下部に）
for role, message in st.session_state.get("message_history", []):
    st.chat_message(role).markdown(message)
