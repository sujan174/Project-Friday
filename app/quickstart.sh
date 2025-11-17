#!/bin/bash

# Aerius Desktop - Quick Start Script
# This script helps you get Aerius Desktop up and running quickly

set -e  # Exit on error

echo "🚀 Aerius Desktop - Quick Start"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Node.js
echo -e "${BLUE}📦 Checking prerequisites...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+ from nodejs.org${NC}"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ Node.js version 18+ required (found: $(node -v))${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js $(node -v) found${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $(python3 --version) found${NC}"

# Check Project-Aerius
echo ""
echo -e "${BLUE}📁 Checking Project-Aerius...${NC}"
if [ ! -d "../Project-Aerius" ]; then
    echo -e "${RED}❌ Project-Aerius not found in parent directory${NC}"
    echo -e "${YELLOW}   Expected location: ../Project-Aerius${NC}"
    exit 1
fi

if [ ! -f "../Project-Aerius/orchestrator.py" ]; then
    echo -e "${RED}❌ orchestrator.py not found in Project-Aerius${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Project-Aerius found${NC}"

# Check .env file
if [ ! -f "../Project-Aerius/.env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found in Project-Aerius${NC}"
    echo -e "${YELLOW}   Creating from .env.example...${NC}"
    if [ -f "../Project-Aerius/.env.example" ]; then
        cp "../Project-Aerius/.env.example" "../Project-Aerius/.env"
        echo -e "${GREEN}✅ .env file created${NC}"
        echo -e "${YELLOW}   Please edit ../Project-Aerius/.env and add your API keys${NC}"
        echo -e "${YELLOW}   Required: GOOGLE_API_KEY${NC}"
        read -p "Press Enter when ready to continue..."
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
fi

# Check for Google API key
if ! grep -q "GOOGLE_API_KEY=.\\+" "../Project-Aerius/.env"; then
    echo -e "${YELLOW}⚠️  GOOGLE_API_KEY not set in .env${NC}"
    echo -e "${YELLOW}   Get your key from: https://makersuite.google.com/app/apikey${NC}"
    read -p "Press Enter when ready to continue..."
fi

# Install npm dependencies
echo ""
echo -e "${BLUE}📦 Installing npm dependencies...${NC}"
if [ ! -d "node_modules" ]; then
    npm install
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# Check Python dependencies
echo ""
echo -e "${BLUE}🐍 Checking Python dependencies...${NC}"
cd ../Project-Aerius
if ! python3 -c "import google.generativeai" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Python dependencies not installed${NC}"
    echo -e "${BLUE}   Installing from requirements.txt...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Python dependencies found${NC}"
fi
cd - > /dev/null

# All checks passed
echo ""
echo -e "${GREEN}✅ All prerequisites met!${NC}"
echo ""
echo -e "${BLUE}🚀 Starting Aerius Desktop...${NC}"
echo ""
echo "This will:"
echo "  1. Start the React development server"
echo "  2. Launch the Electron app"
echo "  3. Initialize the Python backend"
echo ""
echo "First launch may take 30-60 seconds..."
echo ""

# Start the app
npm start
