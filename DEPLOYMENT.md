# Deployment Guide - Retail Sales Forecasting

This guide covers deploying the Retail Sales Forecasting application to various platforms.

---

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for version control)
- Virtual environment tool (venv or conda)

---

## 🚀 Local Deployment

### 1. Clone/Navigate to Project

```bash
cd retail_sales_forecasting
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Data Files

Ensure the following files are in `data/raw/`:
- `train.csv`
- `test.csv`
- `stores.csv`
- `holidays_events.csv`
- `oil.csv`
- `transactions.csv`

### 5. Train Models (Optional)

```bash
# Quick test with subset of data (recommended first)
python -m src.train --quick-test --models baseline gradient_boosting

# Full training (takes 1-2 hours)
python -m src.train
```

**Note**: The dashboard will work without trained models, but Model Performance page will show a warning.

### 6. Run Streamlit Dashboard

```bash
streamlit run Home.py
```

The application will open at `http://localhost:8501`

---

## ☁️ Streamlit Cloud Deployment

### 1. Prepare Repository

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Retail Sales Forecasting"

# Push to GitHub
git remote add origin https://github.com/yourusername/retail-sales-forecasting.git
git push -u origin main
```

### 2. Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path**: `Home.py`
   - **Python version**: 3.9 or 3.10
6. Click "Deploy"

### 3. Handle Large Files

**Option A: Use Git LFS (Large File Storage)**
```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

**Option B: Use External Storage**
- Upload data files to Google Drive, Dropbox, or S3
- Modify `config.py` to download files on startup
- Add download logic to `data_loader.py`

**Option C: Use Streamlit Secrets**
- Store data URLs in `.streamlit/secrets.toml`
- Download on first run

### 4. Optimize for Cloud

Add to `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

---

## 🐳 Docker Deployment

### 1. Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create .dockerignore

```
.venv/
__pycache__/
*.pyc
.git/
.gitignore
*.md
data/raw/*.csv
models/*.pkl
models/*.h5
```

### 3. Build and Run

```bash
# Build image
docker build -t retail-sales-forecasting .

# Run container
docker run -p 8501:8501 retail-sales-forecasting
```

Access at `http://localhost:8501`

### 4. Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## 🌐 AWS Deployment

### Option 1: AWS EC2

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t3.medium (or larger for training)
   - Security group: Allow port 8501

2. **SSH into Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git -y
   ```

4. **Clone and Setup**
   ```bash
   git clone https://github.com/yourusername/retail-sales-forecasting.git
   cd retail-sales-forecasting
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Run with nohup**
   ```bash
   nohup streamlit run Home.py --server.port=8501 --server.address=0.0.0.0 &
   ```

6. **Access**
   - Navigate to `http://your-ec2-ip:8501`

### Option 2: AWS ECS (Elastic Container Service)

1. Push Docker image to ECR
2. Create ECS cluster
3. Define task definition
4. Create service
5. Configure load balancer

---

## 🔧 Production Considerations

### Performance Optimization

1. **Caching**
   ```python
   @st.cache_data
   def load_data():
       # Expensive data loading
       pass
   ```

2. **Lazy Loading**
   - Load models only when needed
   - Sample data for exploration

3. **Async Operations**
   - Use background tasks for training
   - Implement job queue for predictions

### Security

1. **Environment Variables**
   - Store sensitive data in `.env`
   - Use `python-dotenv`

2. **Authentication**
   - Add Streamlit authentication
   - Use OAuth for enterprise

3. **HTTPS**
   - Use reverse proxy (nginx)
   - SSL certificates (Let's Encrypt)

### Monitoring

1. **Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Error Tracking**
   - Sentry integration
   - Custom error handlers

3. **Analytics**
   - Google Analytics
   - Streamlit analytics

---

## 📊 Model Training in Production

### Scheduled Retraining

**Option 1: Cron Job**
```bash
# Add to crontab
0 2 * * 0 cd /path/to/project && .venv/bin/python -m src.train
```

**Option 2: Airflow DAG**
```python
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('retrain_models', schedule_interval='@weekly')

train_task = BashOperator(
    task_id='train_models',
    bash_command='cd /path/to/project && python -m src.train'
)
```

**Option 3: AWS Lambda + EventBridge**
- Trigger training on schedule
- Store models in S3
- Load from S3 in Streamlit

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          python -m pytest tests/
      
      - name: Deploy to Streamlit Cloud
        # Streamlit Cloud auto-deploys on push
        run: echo "Deployed!"
```

---

## 🆘 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   streamlit run Home.py --server.port=8502
   ```

3. **Memory Issues**
   - Use `--quick-test` flag
   - Reduce data sample size
   - Increase instance memory

4. **Model Loading Errors**
   - Ensure models are trained
   - Check file paths in `config.py`
   - Verify model compatibility

---

## 📞 Support

For issues or questions:
- Check the [README.md](README.md)
- Review the [walkthrough.md](walkthrough.md)
- Open an issue on GitHub

---

**Happy Deploying! 🚀**
