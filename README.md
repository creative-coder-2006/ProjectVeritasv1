# 🔍 VERITAS.AI - Truth Detection Platform

VERITAS.AI is an advanced platform for detecting misinformation, deepfakes, and unreliable content across multiple media types using AI-powered analysis.

## 🚀 Features

- **📰 News Article Analysis**: Scrape and analyze news articles for misinformation
- **💬 Reddit Post Analysis**: Analyze Reddit posts and comments for reliability
- **🎬 Video Deepfake Detection**: Detect manipulated video content using computer vision
- **🎵 Audio Transcription & Analysis**: Transcribe and analyze audio content
- **🤝 Trust & Credibility Scoring**: Evaluate content reliability using multiple metrics
- **📊 Explainable AI**: Provide transparent explanations of analysis results
- **🔐 User Authentication**: Secure user registration and login system
- **📈 Results Dashboard**: Comprehensive visualization of analysis results

## 🛠️ Technology Stack

### Core Libraries
- **Streamlit**: Web interface and user experience
- **SQLite**: Local database for user data and analysis results
- **PyTorch**: Deep learning framework for AI models
- **Transformers**: Hugging Face models for text analysis
- **OpenCV**: Computer vision for video analysis
- **Whisper**: OpenAI's speech recognition for audio transcription

### Scraping & Data Collection
- **Newspaper3k**: News article extraction
- **PRAW**: Reddit API wrapper
- **yt-dlp**: YouTube video downloading
- **FFmpeg**: Audio/video processing

### Analysis & ML
- **scikit-learn**: Statistical analysis and metrics
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations

## 📦 Installation

### Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** installed for audio/video processing:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd veritasai
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your API keys:
   ```
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   ```

4. **Initialize the database**:
   ```bash
   python -c "from database import init_database; init_database()"
   ```

## 🚀 Usage

### Starting the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Getting Started

1. **Register/Login**: Create an account or login to access the platform
2. **Enter Topic**: Input a topic you want to analyze (e.g., "climate change", "COVID-19")
3. **Select Content Types**: Choose which types of content to analyze:
   - News Articles
   - Reddit Posts  
   - YouTube Videos
4. **Run Analysis**: Click "Start Analysis" to begin the comprehensive analysis
5. **Review Results**: View detailed results, explanations, and alerts

### Analysis Features

#### Text Analysis
- **Misinformation Detection**: Identify false or misleading information
- **LLM Origin Detection**: Detect AI-generated content
- **Entropy Analysis**: Measure text complexity and naturalness
- **Confidence Scoring**: Assess analysis reliability

#### Video Analysis
- **Deepfake Detection**: Identify manipulated video content
- **Face Anomaly Detection**: Detect inconsistent face patterns
- **Compression Artifact Analysis**: Identify video manipulation artifacts
- **Frame Quality Assessment**: Evaluate video quality metrics

#### Audio Analysis
- **Speech Transcription**: Convert audio to text using Whisper
- **Audio Quality Metrics**: Analyze audio characteristics
- **Anomaly Detection**: Identify audio manipulation
- **Content Analysis**: Analyze transcribed text for misinformation

#### Trust & Credibility
- **Source Reliability**: Evaluate content source reputation
- **Author Credibility**: Assess author trustworthiness
- **Content Quality**: Analyze writing style and structure
- **Temporal Freshness**: Consider content recency
- **Cross-Check Consistency**: Compare with similar content

## 📊 Results Interpretation

### Score Ranges

- **🟢 Low Risk (0.0 - 0.3)**: Content appears reliable
- **🟡 Moderate Risk (0.3 - 0.7)**: Exercise caution
- **🔴 High Risk (0.7 - 1.0)**: Content likely unreliable

### Alert Types

- **Misinformation Alerts**: Indicate potential false information
- **Trust Alerts**: Highlight low reliability sources
- **Deepfake Alerts**: Warn about manipulated video content
- **Credibility Alerts**: Flag questionable content quality

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Reddit API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Model Configuration
MISINFORMATION_MODEL=mrm8488/bert-tiny-fake-news-detection
WHISPER_MODEL=base

# Analysis Limits
MAX_NEWS_ARTICLES=50
MAX_REDDIT_POSTS=100
MAX_VIDEOS=20
```

### Reddit API Setup

1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Note your Client ID and Client Secret
5. Add them to your `.env` file

## 🗄️ Database Schema

The application uses SQLite with the following tables:

- **users**: User authentication and profiles
- **news_articles**: Scraped news content
- **reddit_posts**: Scraped Reddit content  
- **youtube_videos**: Video metadata and analysis
- **analysis_results**: Analysis scores and metrics
- **xai_explanations**: Explainable AI explanations

## 🔒 Privacy & Security

- **Local Processing**: All analysis is performed locally when possible
- **Secure Storage**: User data stored in local SQLite database
- **No Third-Party Sharing**: Content is not shared with external services
- **Automatic Cleanup**: Temporary files are automatically removed
- **Password Hashing**: User passwords are securely hashed using bcrypt

## 🧪 Testing

Run tests to verify installation:

```bash
python -c "
from analyzer.text_analysis import analyze_text
from analyzer.trust_credibility import calculate_trust_score
result = analyze_text('This is a test message.')
print('Text analysis test:', result)
"
```

## 📈 Performance

### Recommended System Requirements

- **CPU**: 4+ cores for parallel processing
- **RAM**: 8GB+ for model loading and analysis
- **Storage**: 10GB+ for models and temporary files
- **GPU**: Optional, but recommended for video analysis

### Optimization Tips

- Use smaller models for faster analysis
- Limit concurrent video downloads
- Enable GPU acceleration if available
- Monitor disk space for temporary files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and libraries
- **OpenAI**: For Whisper speech recognition
- **Streamlit**: For the web framework
- **Reddit**: For API access
- **YouTube**: For video content

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the code comments

---

**🔍 VERITAS.AI - Empowering Truth in the Digital Age** 
