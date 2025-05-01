# 必要なライブラリをインポート
import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import os
import pandas as pd
from io import BytesIO, StringIO

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
        "gemini-1.5-pro-latest": 3.5 / 1_000_000,
        "azure-gpt": 5 / 1_000_000
    },
    "output": {
        "gpt-3.5-turbo": 1.5 / 1_000_000,
        "gpt-4o": 15 / 1_000_000,
        "claude-3-5-sonnet-20240620": 15 / 1_000_000,
        "gemini-1.5-pro-latest": 10.5 / 1_000_000,
        "azure-gpt": 15 / 1_000_000
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
        st.session_state.model_name = "azure-gpt"
        return AzureChatOpenAI(
            temperature=temperature,
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

# チャット用の連携処理をまとめたchainを生成
def init_chain():
    st.session_state.llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        *((entry["Role"], entry["Message"]) for entry in st.session_state.message_history),
        ("user", "{user_input}")
    ])
    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser

# トークン数の計算関数
def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        try:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        except:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

# 利用したトークン数からAPI料金を計算する関数
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

    input_cost = MODEL_PRICES["input"].get(st.session_state.model_name, 0) * input_count
    output_cost = MODEL_PRICES["output"].get(st.session_state.model_name, 0) * output_count

    if "gemini" in st.session_state.model_name and (input_count + output_count) > 128_000:
        input_cost *= 2
        output_cost *= 2

    total_cost = input_cost + output_cost
    st.sidebar.markdown("## 💰 Usage Cost")
    st.sidebar.markdown(f"**Total cost: ${total_cost:.5f}**")
    st.sidebar.markdown(f"- Input: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output: ${output_cost:.5f}")

# 要約とフィードバック関数（LLMによる実行）
def summarize_conversation():
    from langchain_core.prompts import PromptTemplate

    history_text = "\n".join(
        f"{m['Role']}({m['User']}): {m['Message']}" for m in st.session_state.message_history if m['Role'] != 'system'
    )
    summarize_prompt = PromptTemplate.from_template(
        """
        以下はある学習者とAIの会話履歴です。

        {conversation}

        この会話内容を200字以内で要約し、学習の特徴や意欲を踏まえてフィードバックしてください。

        ## 出力形式：
        - 💡要約：
        - 🗣フィードバック：
        """
    )
    summarize_chain = summarize_prompt | st.session_state.llm | StrOutputParser()
    result = summarize_chain.invoke({"conversation": history_text})

    st.subheader("🔚 会話のまとめとフィードバック")
    st.markdown(result)

# アプリのメイン部分
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

    if st.sidebar.button("Save Conversation as Excel"):
        df = pd.DataFrame(st.session_state.message_history)
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        st.sidebar.download_button(
            label="📅 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.sidebar.button("Save Conversation as CSV"):
        df = pd.DataFrame(st.session_state.message_history)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label="📄 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # 会話終了ボタン（要約とフィードバック）
    if st.sidebar.button("🚪 会話を終了して要約とフィードバック"):
        summarize_conversation()

# アプリ起動
if __name__ == "__main__":
    main()
