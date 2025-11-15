#!/bin/bash
# Installation script for Project Aerius

echo "╔════════════════════════════════════════╗"
echo "║   Project Aerius - Installation        ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check Python version
echo "→ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python $python_version detected"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "✗ pip3 not found. Please install pip first."
    exit 1
fi
echo "✓ pip3 found"
echo ""

# Install dependencies
echo "→ Installing dependencies..."
echo ""

# Core dependencies
echo "  Installing core dependencies (Google Gemini, dotenv, numpy)..."
pip3 install -q google-generativeai python-dotenv numpy

# UI dependencies
echo "  Installing UI dependencies (Rich, prompt-toolkit)..."
pip3 install -q rich prompt-toolkit

echo ""
echo "✓ All dependencies installed successfully!"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "→ Setting up environment configuration..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env file from .env.example"
        echo ""
        echo "⚠  IMPORTANT: Please edit .env and add your API keys:"
        echo "   - GOOGLE_API_KEY (required for Gemini)"
        echo "   - NOTION_TOKEN (optional for Notion integration)"
        echo ""
    else
        echo "⚠  No .env.example found. Please create .env manually."
        echo ""
    fi
else
    echo "✓ .env file already exists"
    echo ""
fi

# Create necessary directories
echo "→ Creating necessary directories..."
mkdir -p data logs .cache credentials
echo "✓ Directories created"
echo ""

# Installation complete
echo "╔════════════════════════════════════════╗"
echo "║   Installation Complete! ✨             ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys"
echo "  2. Run: python3 main.py"
echo "  3. For verbose mode: python3 main.py --verbose"
echo "  4. For simple UI: python3 main.py --simple"
echo ""
echo "For help: python3 main.py and type 'help'"
echo ""
