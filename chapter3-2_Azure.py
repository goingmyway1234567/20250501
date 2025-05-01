# 必要なライブラリをインポート
import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import os
import pandas as pd
from io import BytesIO, StringIO

# Azure OpenAI 用のクライアントをインポート
from langchain_openai import AzureChatOpenAI

# .envファイルを使って秘密情報を読み込む（ローカルで開発する場合用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenvが見つかりません。環境変数は手動で設定してください。", ImportWarning)

# モデルごとの料金設定（1トークンあたりのUSD）
MODEL_PRICES = {
    "input": {
        "gpt-3.5-turbo": 0.5 / 1_000_000,
        "gpt-4o": 5 / 1_000_000,
        "claude-3-5-sonnet-20240620": 3 / 1_000_000,
        "gemini-1.5-pro-latest": 3.5 / 1_000_000
    },
    "output": {
        "gpt-3.5-turbo": 1.5 / 1_000_000,
        "gpt-4o": 15 / 1_000_000,
        "claude-3-5-sonnet-20240620": 15 / 1_000_000,
        "gemini-1.5-pro-latest": 10.5 / 1_000_000
    }
}

# ページの基本情報を設定
def init_page():
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")

# 会話履歴を初期化
def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "message_history" not in st.session_state:
        st.session_state.user_name = "mukai"
        st.session_state.message_history = [
            {
                "Role": "system",
                "User": "system",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Message": "You are a helpful assistant."
            }
        ]

# モデルを選ぶ部分（サイドバー）
def select_model():
    temperature = st.sidebar.slider("Temperature:", 0.0, 2.0, 0.0, 0.01)
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Azure OpenAI")
    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
        return ChatOpenAI(
            temperature=temperature,
            model_name=st.session_state.model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(
            temperature=temperature,
            model_name=st.session_state.model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model == "Claude 3.5 Sonnet":
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "Gemini 1.5 Pro":
        st.session_state.model_name = "gemini-1.5-pro-latest"
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "Azure OpenAI":
        st.session_state.model_name = "azure-openai"
        return AzureChatOpenAI(
            temperature=temperature,
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

# 以下の関数は変更ありません（省略）

# アプリ起動
if __name__ == "__main__":
    main()
