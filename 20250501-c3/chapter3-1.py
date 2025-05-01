# 必要なライブラリをインポート
# AIとやりとりするための部品や、画面表示、ファイル保存、日付などを扱うツールを読み込む
import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# .envファイル（環境設定ファイル）を読み込んで、APIキーなどの秘密の情報を使えるようにする
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenvが見つかりません。環境変数は手動で設定してください。", ImportWarning)

# モデルごとの料金を設定（100万トークンあたりの単価）
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

# アプリの見た目を設定する関数
def init_page():
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")

# 会話履歴の初期化関数
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

# モデルを選択する関数
def select_model():
    temperature = st.sidebar.slider("Temperature:", 0.0, 2.0, 0.0, 0.01)
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
        return ChatOpenAI(temperature=temperature, model_name=st.session_state.model_name)
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(temperature=temperature, model_name=st.session_state.model_name)
    elif model == "Claude 3.5 Sonnet":
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(temperature=temperature, model_name=st.session_state.model_name)
    elif model == "Gemini 1.5 Pro":
        st.session_state.model_name = "gemini-1.5-pro-latest"
        return ChatGoogleGenerativeAI(temperature=temperature, model=st.session_state.model_name)

# チャットチェーンの初期化
def init_chain():
    st.session_state.llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        *((entry["Role"], entry["Message"]) for entry in st.session_state.message_history),
        ("user", "{user_input}")
    ])
    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser

# トークン数のカウント
def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        try:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        except:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

# コスト計算と表示
def calc_and_display_costs():
    output_count = 0
    input_count = 0
    for entry in st.session_state.message_history:
        tokens = get_message_counts(entry["Message"])
        if entry["Role"] == "ai":
            output_count += tokens
        else:
            input_count += tokens

    if len(st.session_state.message_history) == 1:
        return

    input_cost = MODEL_PRICES["input"][st.session_state.model_name] * input_count
    output_cost = MODEL_PRICES["output"][st.session_state.model_name] * output_count

    if "gemini" in st.session_state.model_name and (input_count + output_count) > 128_000:
        input_cost *= 2
        output_cost *= 2

    total_cost = input_cost + output_cost
    st.sidebar.markdown("## 💰 Usage Cost")
    st.sidebar.markdown(f"**Total cost: ${total_cost:.5f}**")
    st.sidebar.markdown(f"- Input: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output: ${output_cost:.5f}")

# メイン処理
def main():
    init_page()
    init_messages()
    chain = init_chain()

    for entry in st.session_state.message_history:
        st.chat_message(entry["Role"]).markdown(entry["Message"])

    if user_input := st.chat_input("聞きたいことを入力してね！"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.chat_message("user").markdown(user_input)
        st.session_state.message_history.append({
            "Role": "user",
            "User": st.session_state.user_name,
            "Timestamp": now,
            "Message": user_input
        })
        with st.chat_message("ai"):
            response = st.write_stream(chain.stream({"user_input": user_input}))
        st.session_state.message_history.append({
            "Role": "ai",
            "User": "assistant",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Message": response
        })

    calc_and_display_costs()

    import pandas as pd
    from io import BytesIO, StringIO

    if st.sidebar.button("Save Conversation as Excel"):
        df = pd.DataFrame(st.session_state.message_history)
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_data = excel_buffer.getvalue()
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.sidebar.download_button(
            label="📅 Download Excel",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.sidebar.button("Save Conversation as CSV"):
        df = pd.DataFrame(st.session_state.message_history)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.sidebar.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
