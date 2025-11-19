# 📦 Installation Guide - Project Friday

## 🚀 Quick Start (Current Project)

Install the **minimal dependencies** for the current codebase:

```bash
pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0
```

**OR** use the requirements file:

```bash
pip install -r requirements.txt
```

---

## 🧠 Full ML Implementation (With Continuous Learning)

To implement the **ML Engineering + Data Flywheel** features from the analysis, install:

```bash
# Core ML Stack
pip install torch>=2.1.0 transformers>=4.36.0 scikit-learn>=1.3.0 numpy>=1.24.0 pandas>=2.0.0

# Continuous Learning
pip install schedule>=1.2.0 matplotlib>=3.7.0 seaborn>=0.12.0 tensorboard>=2.15.0

# Plus the core dependencies
pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0
```

**Single command for everything:**

```bash
pip install google-generativeai>=0.8.0 python-dotenv>=1.0.0 mcp>=1.0.0 torch>=2.1.0 transformers>=4.36.0 scikit-learn>=1.3.0 numpy>=1.24.0 pandas>=2.0.0 schedule>=1.2.0 matplotlib>=3.7.0 seaborn>=0.12.0 tensorboard>=2.15.0
```

---

## 🏭 Production-Grade (Optional)

For production deployment with monitoring and distributed caching:

```bash
pip install redis>=5.0.0 prometheus-client>=0.19.0 sentry-sdk>=1.39.0 fastapi>=0.109.0 uvicorn>=0.25.0 pydantic>=2.5.0
```

---

## 📋 Installation by Phase

### **Phase 1-2: Data Collection** (Week 1-2)
```bash
pip install google-generativeai python-dotenv mcp numpy pandas
```

### **Phase 3-4: ML Model Training** (Week 3-6)
```bash
pip install torch transformers scikit-learn numpy pandas matplotlib
```

### **Phase 5-7: Continuous Learning** (Week 6-12)
```bash
pip install schedule tensorboard seaborn
```

---

## 🐳 Docker Installation (Recommended for Production)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the application
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t project-friday .
docker run -it --env-file .env project-friday
```

---

## 🔧 Verify Installation

After installation, verify everything works:

```bash
python -c "import google.generativeai; import dotenv; import mcp; print('✓ Core dependencies installed')"
```

For ML features:
```bash
python -c "import torch; import transformers; import sklearn; print('✓ ML dependencies installed')"
```

---

## 📝 Notes

- **Python Version**: 3.10+ recommended (3.11 preferred)
- **CUDA**: For GPU support with PyTorch, install `torch` with CUDA:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
- **Virtual Environment**: Always use a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  # or
  venv\Scripts\activate  # Windows
  ```

---

## 🆘 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'google.generativeai'`
- **Fix**: `pip install --upgrade google-generativeai`

**Issue**: PyTorch installation fails
- **Fix**: Use conda instead: `conda install pytorch torchvision -c pytorch`

**Issue**: Transformers model download is slow
- **Fix**: Pre-download models: `python -m transformers-cli download distilbert-base-uncased`
