#!/bin/bash
# Project Friday - Installation Commands

echo "=================================================="
echo "  Project Friday - Package Installation"
echo "=================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Prompt user for installation type
echo "Select installation type:"
echo "  1) Minimal (current project only)"
echo "  2) Full ML (with continuous learning)"
echo "  3) Production (with monitoring & caching)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Installing minimal dependencies..."
        pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0
        ;;
    2)
        echo "Installing full ML stack..."
        pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0 \
                    torch>=2.1.0 transformers>=4.36.0 scikit-learn>=1.3.0 \
                    numpy>=1.24.0 pandas>=2.0.0 schedule>=1.2.0 \
                    matplotlib>=3.7.0 seaborn>=0.12.0 tensorboard>=2.15.0
        ;;
    3)
        echo "Installing production stack..."
        pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0 \
                    torch>=2.1.0 transformers>=4.36.0 scikit-learn>=1.3.0 \
                    numpy>=1.24.0 pandas>=2.0.0 schedule>=1.2.0 \
                    matplotlib>=3.7.0 seaborn>=0.12.0 tensorboard>=2.15.0 \
                    redis>=5.0.0 prometheus-client>=0.19.0 sentry-sdk>=1.39.0 \
                    fastapi>=0.109.0 uvicorn>=0.25.0 pydantic>=2.5.0
        ;;
    *)
        echo "Invalid choice. Installing minimal dependencies..."
        pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0
        ;;
esac

echo ""
echo "=================================================="
echo "  ✓ Installation Complete!"
echo "=================================================="
echo ""
echo "Verify installation:"
echo "  python -c \"import google.generativeai; import dotenv; import mcp; print('✓ Ready!')\""
echo ""
