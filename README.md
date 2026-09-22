## What is this?

GoalBlitz AI Coach is a small web app that lets you ask questions about game controls and mechanics (for example, “How do I use Powered Kick?” or “How does Dash work?”), reads your game guide (`GoalBlitz_KnowledgeBase.md`), uses AI to give short, clear coaching tips, and remembers the last few messages so follow-up questions make sense. You open it in your browser and chat with the AI Coach in a side panel.

---

## Installation

1. **Install Ollama**  
   - Go to https://ollama.ai  
   - Download and install Ollama for your system.  
   - Start Ollama (run the app or `ollama serve`).

2. **Pull required models** (once):

   ```bash
   ollama pull llama3.2:3b
   ollama pull embeddinggemma
   ```

3. **Install Python dependencies**  
   In the `GoalBlitzAI` folder, run:

   ```bash
   pip install fastapi uvicorn pydantic requests numpy
   ```

   If `pip` is not found, try:

   ```bash
   python -m pip install fastapi uvicorn pydantic requests numpy
   ```

4. **Check the knowledge base**  
   Make sure this file exists:

   ```text
   knowledge/GoalBlitz_KnowledgeBase.md
   ```

---

## How to use

1. **Start the AI Coach server**

   In the `GoalBlitzAI` folder, run:

   ```bash
   python -m uvicorn rag_server:app --host 127.0.0.1 --port 8000
   ```

   Keep this terminal window open while using the coach.

2. **Open in your browser**

   Go to:

   ```text
   http://127.0.0.1:8000
   ```

3. **Ask questions**

   - In the right-side **AI Coach** panel, type your question and press **Send**.
   - Wait a moment for the AI to answer.
   - You can ask follow-up questions; the AI will use the recent conversation to understand the context.
   - Use the language switcher (EN / 繁 / 简) at the top to change the interface language.
   - You can type questions in English, Traditional Chinese, or Simplified Chinese.

4. **Hide / show the panel**

   - Click the **×** button in the top-right of the AI Coach panel to hide it.
   - Click the floating **AI COACH** button at the bottom-right of the page to show it again.

---

## 這是什麼？

GoalBlitz AI Coach 是一個小型網頁應用程式，可以讓你詢問關於遊戲操作和機制的問題（例如「如何使用強力射門？」或「衝刺怎樣運作？」），讀取你的遊戲指南（`GoalBlitz_KnowledgeBase.md`），使用 AI 提供簡短、清晰的教練提示，並記住最近幾則訊息，讓後續追問的問題也能被正確理解。你在瀏覽器中打開它，並在右側面板與 AI 教練對話。

---

## 安裝

1. **安裝 Ollama**  
   - 前往 https://ollama.ai  
   - 下載並安裝適合你系統的 Ollama。  
   - 啟動 Ollama（執行應用程式或輸入 `ollama serve`）。

2. **下載必要的模型**（只需一次）：

   ```bash
   ollama pull llama3.2:3b
   ollama pull embeddinggemma
   ```

3. **安裝 Python 依賴**  
   在 `GoalBlitzAI` 資料夾中執行：

   ```bash
   pip install fastapi uvicorn pydantic requests numpy
   ```

   如果找不到 `pip`，可以試試：

   ```bash
   python -m pip install fastapi uvicorn pydantic requests numpy
   ```

4. **確認知識庫存在**  
   確保這個檔案存在：

   ```text
   knowledge/GoalBlitz_KnowledgeBase.md
   ```

---

## 如何使用

1. **啟動 AI 教練伺服器**

   在 `GoalBlitzAI` 資料夾中執行：

   ```bash
   python -m uvicorn rag_server:app --host 127.0.0.1 --port 8000
   ```

   使用教練時請保持這個終端視窗開啟。

2. **在瀏覽器中打開**

   前往：

   ```text
   http://127.0.0.1:8000
   ```

3. **提出問題**

   - 在右側的 **AI 教練** 面板輸入你的問題，然後按下 **送出**。
   - 稍等片刻，AI 會給出回答。
   - 你可以繼續追問；AI 會根據最近的對話內容來理解你在問什麼。
   - 使用頂部的語言切換器（EN / 繁 / 简）切換介面語言。
   - 你可以用英文、繁體中文或簡體中文輸入問題。

4. **隱藏 / 顯示面板**

   - 點擊 AI 教練面板右上角的 **×** 按鈕即可隱藏面板。
   - 點擊頁面右下角的浮動 **AI COACH** 按鈕即可再次顯示。
