# 必要なライブラリをインポート
import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# モデルを読み込む（OpenAI、Anthropic、Google）
from langchain_openai import ChatOpenAI


import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os

# .envファイルの読み込み（ローカル開発用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenvが見つかりません。環境変数は手動で設定してください。", ImportWarning)

# 要約プロンプトのテンプレート
SUMMARIZE_PROMPT = """以下のコンテンツについて、内容を300文字程度でわかりやすく要約してください。

========

{content}

========

日本語で書いてね！
"""

# Streamlit ページ初期化

def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")

# モデル選択

def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )


# チェーン初期化

def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain

# URL の検証

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Web ページの本文抽出

def get_content(url):
    try:
        with st.spinner("Fetching Website ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write(traceback.format_exc())
        return None

# メイン関数

def main():
    init_page()
    chain = init_chain()

    # URL 入力フィールド
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.error('正しいURLを入力してください。')
        else:
            if content := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

if __name__ == '__main__':
    main()
