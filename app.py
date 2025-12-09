import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# --- 頁面設定 ---
st.set_page_config(page_title="DataInsight: 可解釋化數據助理 (Gemini版)", layout="wide", page_icon="📊")

st.title("📊 DataInsight: 交通/工業數據可解釋化助理")
st.markdown("**結合大數據分析流程與 Gemini API，自動生成數據洞察報告。**")

# --- Sidebar ---
with st.sidebar:
    st.header("設定")
    # 這裡提示使用者輸入 Google API Key
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.caption("請至 Google AI Studio 申請免費 Key")
    
    domain = st.selectbox("選擇應用場域", ["智慧交通 (Traffic)", "智慧工廠 (Factory)"])
    st.info("💡 此系統模擬數據工程師的分析流程：清洗 -> 統計 -> 解釋")

# --- 模擬數據生成 ---
def get_mock_data(domain):
    if domain == "智慧交通 (Traffic)":
        data = {
            'Time': ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00'],
            'Flow (veh/hr)': [1200, 800, 400, 350, 450, 400, 380],
            'Speed (km/h)': [15, 30, 60, 65, 60, 62, 65],
            'Occupancy (%)': [85, 60, 20, 15, 25, 20, 18]
        }
    else: # 工廠
        data = {
            'Time': ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00'],
            'Temperature (C)': [60, 65, 85, 90, 70, 65, 62],
            'Vibration (mm/s)': [0.5, 0.6, 2.8, 3.1, 0.8, 0.6, 0.5],
            'Output (units)': [100, 98, 40, 35, 95, 100, 100]
        }
    return pd.DataFrame(data)

# --- 核心分析與 LLM 生成 (Gemini 版本) ---
def generate_insight(df, domain, api_key):
    if not api_key:
        return "⚠️ 請輸入 Google API Key 以獲取 AI 深度解讀報告。"
    
    # 設定 Gemini API
    try:
        genai.configure(api_key=api_key)
        # 使用 gemini-1.0-pro 模型
        model = genai.GenerativeModel('gemini-flash-latest')
    except Exception as e:
        return f"API 設定錯誤: {str(e)}"
    
    # 1. 簡易統計特徵 (Data Engineering Part)
    stats = df.describe().to_string()
    
    # 2. 建構 Prompt (XAI Part)
    if domain == "智慧交通 (Traffic)":
        role_prompt = "你是一位資深的交通數據分析師。"
        task_prompt = f"""
        {role_prompt}
        請分析以下交通數據統計值：
        {stats}
        
        重點觀察：
        1. 車流 (Flow) 與 車速 (Speed) 的關係。
        2. 是否有壅塞發生？(提示：低速、高佔有率)
        3. 給出 3 點交通疏導建議。
        請用繁體中文回答。
        """
    else:
        role_prompt = "你是一位資深的工廠設備維運工程師。"
        task_prompt = f"""
        {role_prompt}
        請分析以下機台感測數據統計值：
        {stats}
        
        重點觀察：
        1. 溫度 (Temperature) 與 震動 (Vibration) 是否有異常飆高？
        2. 產量 (Output) 是否受到影響？
        3. 給出 3 點設備維護建議。
        請用繁體中文回答。
        """

    try:
        # 呼叫 Gemini 生成內容
        response = model.generate_content(task_prompt)
        
        if not response.parts:
            # The user might have modified the file in the meantime, so I am being more careful
            # about what I assume about the response object.
            try:
                feedback = response.prompt_feedback
                block_reason = feedback.block_reason.name if feedback and hasattr(feedback, 'block_reason') and feedback.block_reason else "未提供"
                
                finish_reason = "未知"
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason.name
                
                return f"生成內容被阻擋或為空。結束原因: {finish_reason}。阻擋原因: {block_reason}。請檢查提示詞或安全設定。"
            except Exception:
                # Fallback for unexpected response structure
                return "生成內容為空，且無法讀取詳細的回饋資訊。請檢查 API 金鑰與模型權限。"

        return response.text
    except Exception as e:
        return f"生成錯誤: {str(e)}"

# --- 主介面佈局 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 數據輸入與視覺化")
    uploaded_file = st.file_uploader("上傳 CSV (時間, 數值A, 數值B...)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = get_mock_data(domain)
        st.caption("目前使用範例數據 (可自行上傳 CSV 替換)")
    
    # 簡單的資料清洗展示
    st.dataframe(df.style.highlight_max(axis=0, color='lightcoral'))
    
    # 繪圖
    fig, ax = plt.subplots()
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 2:
        sns.lineplot(data=df, x='Time', y=numeric_cols[0], ax=ax, label=numeric_cols[0])
        ax2 = ax.twinx()
        sns.lineplot(data=df, x='Time', y=numeric_cols[1], ax=ax2, color='orange', label=numeric_cols[1])
        st.pyplot(fig)

with col2:
    st.subheader("2. AI 診斷報告 (Gemini XAI)")
    if st.button("🔍 開始分析數據"):
        with st.spinner("Gemini 正在解讀數據趨勢..."):
            report = generate_insight(df, domain, api_key)
            st.markdown("### 📋 分析結果")
            st.write(report)
            
            st.success("報告生成完成！")

# --- Footer ---
st.markdown("---")
st.caption("Lab Project: Industrial & Traffic Data Analysis | Powered by Streamlit & Google Gemini")