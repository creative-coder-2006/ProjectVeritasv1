# VERITAS.AI - Comprehensive Misinformation Detection Platform

🔍 **VERITAS.AI** is a comprehensive misinformation detection platform that analyzes content across multiple platforms including news articles, Reddit posts/comments, and YouTube videos to detect misinformation, LLM-generated content, and coordinated influence operations.

## 🚀 Features

### 🔍 Multi-Platform Analysis
- **News Articles**: Analyze news content for misinformation and credibility
- **Reddit Posts & Comments**: Comprehensive Reddit analysis with subreddit flagging
- **YouTube Videos**: Video analysis with audio transcription and frame analysis

### 🧠 Advanced AI Detection
- **Misinformation Detection**: Using BERT-based models for accurate detection
- **LLM Origin Detection**: Detect AI-generated content using statistical and model-based approaches
- **Deepfake Detection**: Video frame analysis for visual manipulation detection
- **Coordinated Operations**: Cross-platform coordination pattern detection

### 📊 Comprehensive Scoring
- **Misinformation Score**: Probability of content being misinformation
- **Confidence Score**: Model confidence in the analysis
- **Trust Score**: Overall trustworthiness assessment
- **Credibility Score**: Source and content credibility evaluation
- **LLM Origin Probability**: Likelihood of AI-generated content

### 🎯 Platform-Specific Features

#### Reddit Analysis
- Post and comment analysis
- Subreddit-level flagging (flags subreddits with multiple high-misinfo posts)
- User engagement analysis
- Suspicious pattern detection
- Subreddit credibility scoring

#### YouTube Analysis
- Audio transcription using OpenAI Whisper
- Video frame extraction and analysis
- Deepfake detection with R3D_18 model
- Channel-level flagging
- Visual manipulation detection
- Temporal consistency analysis

#### News Analysis
- Source reputation tracking
- Author credibility assessment
- Cross-reference analysis
- Fact-checking integration

### 🤖 Explainable AI (XAI)
- **Master Prompts**: Platform-specific explanation generation
- **OpenAI Integration**: AI-powered explanations for suspicious content
- **Feature Importance**: Detailed analysis of what factors contribute to scores
- **Risk Assessment**: Clear risk levels and recommendations

### 📈 Dashboard & Visualization
- **History Dashboard**: Track all analysis runs over time
- **Real-time Analysis**: Live analysis of content
- **Comprehensive Reports**: Detailed reports with recommendations
- **Filtering & Search**: Advanced filtering by platform, source, date, and scores

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/veritasai.git
cd veritasai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
# Reddit API (required for Reddit analysis)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# OpenAI API (optional, for enhanced explanations)
OPENAI_API_KEY=your_openai_api_key
```

4. **Initialize the database**
```bash
python -c "from database import init_database; init_database()"
```

## 🚀 Quick Start

### Basic Usage

```python
from analyzer.pipeline import run_comprehensive_analysis
from scraper.reddit_scraper import fetch_comprehensive_reddit_data
from scraper.video_scraper import fetch_comprehensive_video_data

# Fetch data
topic = "artificial intelligence"
reddit_data = fetch_comprehensive_reddit_data(topic, limit=20)
youtube_data = fetch_comprehensive_video_data(topic, max_results=10)

# Run comprehensive analysis
report = run_comprehensive_analysis(
    topic=topic,
    reddit_data=reddit_data,
    youtube_data=youtube_data
)

# View results
print(f"Risk Level: {report['summary_statistics']['overall_risk_level']}")
print(f"Flagged Content: {report['summary_statistics']['total_flagged_content']}")
```

### Individual Component Usage

```python
from analyzer.text_analysis import comprehensive_text_analysis
from analyzer.xai_explanations import generate_comprehensive_explanation_enhanced

# Analyze text
text = "Your content here..."
analysis = comprehensive_text_analysis(text, source="Test Source", content_type="news")

# Generate explanation
explanation = generate_comprehensive_explanation_enhanced(
    content=text,
    misinfo_score=analysis['misinformation_score'],
    confidence_score=analysis['confidence_score'],
    llm_prob=analysis['llm_origin_probability'],
    trust_score=analysis['trust_score'],
    credibility_score=analysis['credibility_score'],
    source="Test Source",
    content_type="news"
)
```

## 🧪 Testing

The application includes built-in testing capabilities through the Streamlit interface. You can test individual components and run comprehensive analysis directly through the web application.

## 📊 Configuration

### Analysis Thresholds
Configure detection thresholds in `config.py`:

```python
# Misinformation detection thresholds
MISINFO_THRESHOLD = 0.7  # Flag content with >70% misinfo probability
LLM_DETECTION_THRESHOLD = 0.7  # Flag content with >70% LLM probability

# Platform flagging thresholds
SUBREDDIT_FLAG_THRESHOLD = 5  # Flag subreddit after 5 high-misinfo posts
CHANNEL_FLAG_THRESHOLD = 3  # Flag channel after 3 high-misinfo videos
```

### Model Configuration
```python
# Model settings
MISINFORMATION_MODEL = "mrm8488/bert-tiny-fake-news-detection"
LLM_ORIGIN_MODEL = "microsoft/DialoGPT-medium"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
VIDEO_ANALYSIS_MODEL = "r3d_18"
```

## 📁 Project Structure

```
veritasai/
├── analyzer/                 # Analysis modules
│   ├── text_analysis.py     # Text-based misinformation detection
│   ├── video_analysis.py    # Video and deepfake analysis
│   ├── audio_analysis.py    # Audio transcription and analysis
│   ├── trust_credibility.py # Trust and credibility scoring
│   ├── xai_explanations.py  # Explainable AI explanations
│   └── pipeline.py          # Main analysis pipeline
├── scraper/                 # Data collection modules
│   ├── reddit_scraper.py    # Reddit data collection
│   ├── video_scraper.py     # YouTube data collection
│   └── news_scraper.py      # News data collection
├── database.py              # Database operations
├── config.py                # Configuration settings
├── app.py                   # Streamlit web application

└── requirements.txt         # Dependencies
```

## 🔧 Advanced Usage

### Custom Analysis Pipeline

```python
from analyzer.pipeline import MisinformationDetectionPipeline

# Create custom pipeline
pipeline = MisinformationDetectionPipeline()

# Analyze specific platforms
reddit_analysis = pipeline.analyze_reddit_content(reddit_data)
youtube_analysis = pipeline.analyze_youtube_content(youtube_data)

# Detect coordinated operations
coordination = pipeline.detect_coordinated_operations({
    'reddit': reddit_analysis,
    'youtube': youtube_analysis
}, topic="your_topic")

# Generate comprehensive report
report = pipeline.generate_comprehensive_report({
    'reddit': reddit_analysis,
    'youtube': youtube_analysis,
    'coordination': coordination
}, topic="your_topic")
```

### Database Integration

```python
from database import (
    store_reddit_posts, store_youtube_videos,
    store_analysis_result, get_analysis_history
)

# Store data
store_reddit_posts(posts, user_id=1)
store_youtube_videos(videos, user_id=1)

# Store analysis results
store_analysis_result(content_id, "reddit", analysis_data, user_id=1)

# Retrieve history
history = get_analysis_history(user_id=1, limit=50)
```

## 📈 Performance Optimization

### GPU Acceleration
For faster processing, ensure CUDA is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Batch Processing
For large datasets, use batch processing:
```python
from analyzer.text_analysis import analyze_multiple_texts

texts = ["text1", "text2", "text3", ...]
analyses = analyze_multiple_texts(texts)
```

### Rate Limiting
Configure rate limits in `config.py`:
```python
RATE_LIMITS = {
    "reddit_requests_per_minute": 60,
    "youtube_requests_per_minute": 30,
    "openai_requests_per_minute": 20
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BERT Models**: Hugging Face Transformers
- **Whisper**: OpenAI for audio transcription
- **R3D_18**: Facebook Research for video analysis
- **PRAW**: Reddit API wrapper
- **yt-dlp**: YouTube data extraction

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

## 🔮 Roadmap

- [ ] Real-time streaming analysis
- [ ] Additional language support
- [ ] Mobile application
- [ ] API endpoints
- [ ] Advanced visualization dashboard
- [ ] Machine learning model training interface
- [ ] Integration with fact-checking APIs
- [ ] Social media platform expansion

---

**VERITAS.AI** - Empowering truth in the digital age 🔍✨ 