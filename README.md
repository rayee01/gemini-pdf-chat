# Gemini PDF Chat

A Streamlit app for chatting with your PDFs using **Google Gemini** + **LangChain** + **FAISS**. Upload one or more PDFs, and ask natural-language questions about their content — answers are grounded only in the text retrieved from your documents.

The repo also includes a few smaller standalone Gemini demos built while putting this together.

## Features

- 📚 **Chat with multiple PDFs** (`app3.py`) — extracts text, chunks it, embeds it with Gemini embeddings, stores it in a local FAISS index, and answers questions using retrieval-augmented generation (RAG).
- 💬 **Simple Q&A chatbot** (`qachat.py`) — streaming chat with Gemini and in-session chat history.
- 🤖 **Basic Gemini prompt app** (`app.py`) — minimal single-turn question/answer demo.
- 🧾 **Invoice extractor** (`app2.py`) — Gemini Vision demo for pulling information out of an uploaded invoice image.

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://python.langchain.com/) (`langchain`, `langchain_google_genai`, `langchain_community`) — text splitting, embeddings, retrieval chain
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [PyPDF2](https://pypi.org/project/PyPDF2/) — PDF text extraction
- [Google Generative AI](https://ai.google.dev/) (Gemini) — LLM + embeddings
- [python-dotenv](https://pypi.org/project/python-dotenv/) — environment variable loading

## Project Structure

```
.
├── app.py              # Minimal Gemini Q&A demo
├── app2.py              # Invoice image extractor (Gemini Vision)
├── app3.py               # Main app: chat with multiple PDFs (RAG + FAISS)
├── qachat.py             # Streaming chat demo with session history
├── faiss_index/          # Saved FAISS vector store (generated at runtime)
├── requirements.txt
└── .env                  # Your API key (not committed — see below)
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/gemini-pdf-chat.git
   cd gemini-pdf-chat
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Gemini API key**

   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
   Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey). **Never commit this file** — it's already excluded via `.gitignore`.

## Usage

Run whichever app you want:

```bash
# Chat with multiple PDFs (main app)
streamlit run app3.py

# Streaming Q&A chatbot
streamlit run qachat.py

# Basic single-turn Q&A
streamlit run app.py

# Invoice extractor
streamlit run app2.py
```

For `app3.py`: upload one or more PDFs in the sidebar, click **Submit and Process** to build the FAISS index, then ask questions in the main text box.

## Notes

- The FAISS index is saved locally to `faiss_index/` and rebuilt each time you reprocess PDFs.
- `app2.py` references `gemini-pro-vision`, which is deprecated — swap in a current Gemini vision-capable model (check [available models](https://ai.google.dev/gemini-api/docs/models)) before relying on it.
- If a model name (e.g. `gemini-3.6-flash`) returns a 404, check the [current model list](https://ai.google.dev/gemini-api/docs/models) for your API version and update it in the relevant file.

## License

Add a license of your choice (e.g. MIT) if you plan to share this publicly.
