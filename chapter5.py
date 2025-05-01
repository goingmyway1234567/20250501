# Github: https://github.com/naotaka1128/llm_app_codes/chapter05/part1/main.py

import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


SUMMARIZE_PROMPT = """以下のコンテンツについて、内容を300文字程度でわかりやすく要約してください。

========

{content}

========

日本語で書いてね！
"""


def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo"
        )
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o"
        )
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=temperature,
            model_name="claude-3-5-sonnet-20240620"
        )
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-1.5-pro-latest"
        )


def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        ("user", SUMMARIZE_PROMPT),
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


def validate_url(url):
    """ URLが有効かどうかを判定する関数 """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    try:
        with st.spinner("Fetching Website ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # なるべく本文の可能性が高い要素を取得する
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write(traceback.format_exc())  # エラーが発生した場合はエラー内容を表示
        return None


def main():
    init_page()
    chain = init_chain()

    # ユーザーの入力を監視
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
        else:
            if content := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

    # コストを表示する場合は第3章と同じ実装を追加してください
    # calc_and_display_costs()


if __name__ == '__main__':
    main()