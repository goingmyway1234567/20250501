# 必要なライブラリをインポート
import os
import traceback
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# .envファイルを読み込む（ローカル用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenvが見つかりません。環境変数は手動で設定してください。", ImportWarning)

# 要約用プロンプト
SUMMARIZE_PROMPT = """以下のコンテンツについて、内容を300文字程度でわかりやすく要約してください。

========

{content}

========

日本語で書いてね！"""

# Streamlitページ初期設定
def init_page():
    st.set_page_config(page_title="Website Summarizer", page_icon="🤗")
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")

# モデル選択関数
def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.sidebar.error("OPENAI_API_KEY が設定されていません！")
        st.stop()

    if model == "GPT-3.5":
        return ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    elif model == "GPT-4":
        return ChatOpenAI(temperature=temperature, model_name="gpt-4o", openai_api_key=api_key)
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(temperature=temperature, model_name="claude-3-5-sonnet-20240620")
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-1.5-pro-latest")

# チェーン生成関数
def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

# URL検証関数
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# URLの本文抽出関数
def get_content(url):
    try:
        with st.spinner("Fetching Website ..."):
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.error("サイトの取得中にエラーが発生しました。")
        st.text(traceback.format_exc())
        return None

# メイン処理
def main():
    init_page()
    chain = init_chain()

    url = st.text_input("URLを入力してください:")
    if url:
        if not validate_url(url):
            st.warning("有効なURLを入力してください。")
            return

        content = get_content(url)
        if content:
            st.markdown("## 🔍 要約結果")
            with st.spinner("要約中..."):
                try:
                    result = chain.invoke({"content": content})
                    st.success(result)
                except Exception as e:
                    st.error("要約に失敗しました。")
                    st.text(str(e))

            st.markdown("---")
            st.markdown("## 📄 元の本文")
            st.text_area("Extracted Content", content, height=300)

if __name__ == '__main__':
    main()
