# 🤖 Local LLM Runner

A user-friendly desktop application that allows anyone to run AI language models locally on their computer. No coding experience required!

## ✨ Features

- **🔍 Browse & Search**: Search thousands of AI models from HuggingFace
- **⬇️ Easy Downloads**: One-click model downloads directly from HuggingFace
- **💬 Chat Interface**: Simple chat interface to interact with your models
- **📊 Model Management**: View, load, and delete models easily
- **🖥️ Local Execution**: Run models completely offline on your own hardware
- **🎨 Beautiful UI**: Clean, modern web interface designed for non-technical users

## 🚀 Quick Start

### Prerequisites

You need to have Python 3.8 or higher installed on your computer.

**Check if you have Python:**
```bash
python --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/)

### Installation

1. **Download this repository** (or clone it if you know how)

2. **Open a terminal/command prompt** in the project folder

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install all necessary packages. It may take a few minutes.

### Running the Application

Simply run:
```bash
python run.py
```

The application will:
- Start a local server
- Automatically open in your web browser at `http://localhost:5000`

## 📖 How to Use

### 1. Search for Models

- Click on the **"Browse Models"** tab
- Type a model name in the search box (try "gpt2" or "TinyLlama")
- Click **Search**

**Recommended models for beginners:**
- **gpt2** - Small, fast, good for testing (548 MB)
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - Great for chat (2.2 GB)
- **microsoft/phi-2** - Powerful small model (5.5 GB)

### 2. Download a Model

- Click the **Download** button on any model
- Wait for the download to complete (this can take several minutes depending on model size)
- Downloaded models appear in the **"My Models"** tab

### 3. Load a Model

- Go to the **"My Models"** tab
- Click **Load** on the model you want to use
- Wait for the model to load (you'll see a success message)

### 4. Chat with the Model

- Go to the **"Chat"** tab
- Type your message in the text box
- Click **Send** or press Enter
- Wait for the model to generate a response

### 5. Adjust Settings

You can customize the model's behavior:
- **Max Length**: How long the response should be
- **Temperature**: Higher = more creative, Lower = more focused

## 💾 System Requirements

**Minimum:**
- 8 GB RAM
- 10 GB free disk space
- Modern CPU

**Recommended:**
- 16 GB RAM or more
- NVIDIA GPU with CUDA support (for faster inference)
- 50 GB free disk space

**Note:** Smaller models like GPT2 can run on modest hardware. Larger models require more RAM and benefit greatly from a GPU.

## 🔧 Troubleshooting

### "Module not found" errors
Run: `pip install -r requirements.txt`

### Server won't start
Make sure port 5000 is not being used by another application.

### Model download fails
- Check your internet connection
- Make sure you have enough disk space
- Some models may require HuggingFace authentication (create a free account)

### Out of memory errors
- Try a smaller model (like GPT2)
- Close other applications
- Reduce the max length setting

## 📁 Project Structure

```
local-llm/
├── app/
│   ├── backend/
│   │   ├── server.py          # Flask web server
│   │   └── model_manager.py   # Model download & inference
│   └── frontend/
│       ├── index.html         # Main UI
│       ├── styles.css         # Styling
│       └── app.js             # Frontend logic
├── models/                    # Downloaded models stored here
├── requirements.txt           # Python dependencies
├── run.py                     # Main entry point
└── README.md                  # This file
```

## 🛡️ Privacy & Security

- **100% Local**: All models run on your computer
- **No Data Collection**: Your conversations are never sent to external servers
- **Offline Mode**: Works completely offline after downloading models

## 📚 Popular Models to Try

| Model | Size | Best For | Difficulty |
|-------|------|----------|------------|
| gpt2 | 548 MB | Testing, simple tasks | Beginner |
| TinyLlama-1.1B-Chat | 2.2 GB | Chat, Q&A | Beginner |
| microsoft/phi-2 | 5.5 GB | Coding, reasoning | Intermediate |
| mistralai/Mistral-7B-v0.1 | 14 GB | Advanced tasks | Advanced (GPU recommended) |

## 🤝 Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Make sure you're using Python 3.8+
3. Verify all dependencies are installed
4. Check that you have sufficient disk space and RAM

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Models from [HuggingFace](https://huggingface.co/)
- Built with [Flask](https://flask.palletsprojects.com/) and [Transformers](https://huggingface.co/transformers/)

---

**Enjoy running AI models locally! 🎉**
