#!/usr/bin/env python3
"""
Local LLM Runner - Main Entry Point

This script starts the Local LLM Runner application.
Simply run: python run.py
"""

import os
import sys
import webbrowser
from pathlib import Path
import time
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import transformers
        import torch
        import huggingface_hub
        return True
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\n📦 Please install dependencies first:")
        print("   pip install -r requirements.txt\n")
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Local LLM Runner")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    print("✅ All dependencies installed")
    print()

    # Change to backend directory
    backend_dir = Path(__file__).parent / "app" / "backend"
    os.chdir(backend_dir)

    print("🚀 Starting server...")
    print()
    print("Server will be available at: http://localhost:5000")
    print()
    print("📖 Quick Start:")
    print("   1. Search for models (try 'gpt2' or 'TinyLlama')")
    print("   2. Download a model")
    print("   3. Load the model")
    print("   4. Start chatting!")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Give the user a moment to read
    time.sleep(2)

    # Open browser
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass

    # Start the Flask server
    try:
        from server import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        print("Thank you for using Local LLM Runner!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
