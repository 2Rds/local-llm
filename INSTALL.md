# Installation Guide

This guide will help you install and run Local LLM Runner on your computer.

## Step-by-Step Installation

### For Windows Users

1. **Install Python**
   - Download Python from: https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation
   - Click "Install Now"

2. **Download this project**
   - Click the green "Code" button on GitHub
   - Select "Download ZIP"
   - Extract the ZIP file to your desired location

3. **Open Command Prompt**
   - Press `Windows + R`
   - Type `cmd` and press Enter
   - Navigate to the project folder:
     ```
     cd C:\path\to\local-llm
     ```

4. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   (This may take 5-10 minutes)

5. **Run the application**
   ```
   python run.py
   ```

### For Mac Users

1. **Install Python** (if not already installed)
   - Download from: https://www.python.org/downloads/
   - Or use Homebrew: `brew install python`

2. **Download this project**
   - Click the green "Code" button on GitHub
   - Select "Download ZIP"
   - Extract the ZIP file

3. **Open Terminal**
   - Press `Cmd + Space`
   - Type "Terminal" and press Enter
   - Navigate to the project folder:
     ```
     cd ~/Downloads/local-llm
     ```

4. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python3 run.py
   ```

### For Linux Users

1. **Ensure Python 3.8+ is installed**
   ```bash
   python3 --version
   ```

2. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd local-llm
   ```

3. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python3 run.py
   ```

## Verification

If everything is installed correctly, you should see:
```
🤖 Local LLM Runner
Server will be available at: http://localhost:5000
```

And your web browser should automatically open the application.

## Common Issues

### "Python is not recognized"
- Reinstall Python and make sure to check "Add Python to PATH"
- Or add Python to your PATH manually

### "pip is not recognized"
- Try `python -m pip install -r requirements.txt`
- Or `python3 -m pip install -r requirements.txt`

### Permission errors on Mac/Linux
- Use: `pip3 install --user -r requirements.txt`

### Port 5000 already in use
- Another application is using port 5000
- Close that application or modify the port in `run.py`

## Getting Help

If you're still having trouble:
1. Make sure you're using Python 3.8 or higher
2. Check that you have at least 10GB of free disk space
3. Ensure you have a stable internet connection for downloading models

## Next Steps

Once installed, check out the main README.md for:
- How to search and download models
- How to chat with models
- Recommended models for beginners
