import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
# In your main application file (e.g., main.py)

from scraper.reddit_scraper import fetch_reddit_posts
from scraper.news_scraper import fetch_news_articles
from scraper.video_scraper import fetch_youtube_videos

def run_analysis(topic):
    # ... other code
    try:
        reddit_df = fetch_reddit_posts(topic)
        # ... process reddit_df
    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
    # ... other scraper calls

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import modules
from auth import register_user, login_user
from database import init_database, store_news_articles, store_misinfo_analysis, update_news_source_stats, get_news_with_analysis, get_flagged_sources, get_analysis_history, update_subreddit_stats, update_channel_stats
import utils

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="VERITAS.AI - Truth Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .alert-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef6c00;
    }
    .alert-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Lazy loading functions for heavy modules
def get_text_analyzer():
    """Get text analyzer with lazy import."""
    try:
        from analyzer.text_analysis import comprehensive_text_analysis, analyze_multiple_texts
        return comprehensive_text_analysis, analyze_multiple_texts
    except Exception as e:
        st.error(f"Error loading text analyzer: {e}")
        return None, None

def get_trust_analyzer():
    """Get trust analyzer with lazy import."""
    try:
        from analyzer.trust_credibility import calculate_trust_score, calculate_credibility_score
        return calculate_trust_score, calculate_credibility_score
    except Exception as e:
        st.error(f"Error loading trust analyzer: {e}")
        return None, None

def get_xai_explainer():
    """Get XAI explainer with lazy import."""
    try:
        from analyzer.xai_explanations import generate_comprehensive_explanation, format_explanation_for_display
        return generate_comprehensive_explanation, format_explanation_for_display
    except Exception as e:
        st.error(f"Error loading XAI explainer: {e}")
        return None, None

def get_scrapers():
    """Get scrapers with lazy import."""
    try:
        from scraper.news_scraper import fetch_news
        from scraper.reddit_scraper import fetch_reddit_posts
        from scraper.video_scraper import fetch_videos, download_videos_for_analysis
        return fetch_news, fetch_reddit_posts, fetch_videos, download_videos_for_analysis
    except Exception as e:
        st.error(f"Error loading scrapers: {e}")
        return None, None, None, None

# Session state management
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

def main():
    """Main application function."""
    
    # Sidebar for authentication
    with st.sidebar:
        st.title("🔐 Authentication")
        
        if st.session_state.user_id is None:
            # Login/Register form
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                st.subheader("Login")
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("Login"):
                    if login_username and login_password:
                        success, result = login_user(login_username, login_password)
                        if success:
                            st.session_state.user_id = result['user_id']
                            st.session_state.username = result['username']
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error(result)
                    else:
                        st.error("Please enter username and password")
            
            with tab2:
                st.subheader("Register")
                reg_username = st.text_input("Username", key="reg_username")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
                
                if st.button("Register"):
                    if reg_username and reg_password and reg_confirm:
                        if reg_password == reg_confirm:
                            success, message = register_user(reg_username, reg_password)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.error("Please fill all fields")
        else:
            # User is logged in
            st.success(f"Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()
    
    # Main content area
    if st.session_state.user_id is None:
        # Show welcome page for non-authenticated users
        st.markdown('<h1 class="main-header">🔍 VERITAS.AI</h1>', unsafe_allow_html=True)
        st.markdown("""
        ### Truth Detection Platform
        
        VERITAS.AI is an advanced platform for detecting misinformation, deepfakes, and unreliable content across multiple media types.
        
        **Features:**
        - 📰 News article analysis
        - 💬 Reddit post analysis  
        - 🎬 Video deepfake detection
        - 🎵 Audio transcription and analysis
        - 🤝 Trust and credibility scoring
        - 📊 Explainable AI explanations
        
        Please login or register to start analyzing content.
        """)
        
    else:
        # User is authenticated - show main application
        st.markdown('<h1 class="main-header">🔍 VERITAS.AI</h1>', unsafe_allow_html=True)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📈 Results", "📚 History", "ℹ️ About"])
        
        with tab1:
            show_analysis_tab()
        
        with tab2:
            show_results_tab()
        
        with tab3:
            show_history_tab()
        
        with tab4:
            show_about_tab()

def show_analysis_tab():
    """Show the main analysis tab."""
    st.subheader("Content Analysis")
    
    # Topic input
    topic = st.text_input("Enter a topic to analyze:", placeholder="e.g., climate change, COVID-19, elections")
    
    if topic:
        # Content type selection
        content_types = st.multiselect(
            "Select content types to analyze:",
            ["News Articles", "Reddit Posts", "YouTube Videos"],
            default=["News Articles"]
        )
        
        if st.button("🔍 Start Analysis", type="primary"):
            with st.spinner("Analyzing content..."):
                results = perform_comprehensive_analysis(topic, content_types)
                st.session_state.analysis_results = results
                st.success("Analysis completed!")
                st.rerun()

def perform_comprehensive_analysis(topic, content_types):
    """Perform comprehensive analysis of content with enhanced features."""
    
    results = {
        'topic': topic,
        'timestamp': datetime.now(),
        'content_types': content_types,
        'news_articles': [],
        'reddit_posts': [],
        'videos': [],
        'analysis_results': [],
        'coordinated_operations': None
    }
    
    try:
        # Load analyzers and scrapers lazily
        comprehensive_text_analysis, analyze_multiple_texts = get_text_analyzer()
        generate_comprehensive_explanation, format_explanation_for_display = get_xai_explainer()
        fetch_news, fetch_reddit_posts, fetch_videos, download_videos_for_analysis = get_scrapers()
        
        # Check if analyzers loaded successfully
        if not comprehensive_text_analysis or not generate_comprehensive_explanation:
            st.error("Failed to load analysis modules. Please check your installation.")
            return results
        
        # Fetch news articles
        if "News Articles" in content_types:
            utils.log_analysis_step("Fetching news articles", {"topic": topic})
            if fetch_news:
                news_articles = fetch_news(topic, max_articles=10)
                results['news_articles'] = news_articles
                
                # Store articles in database
                if news_articles:
                    stored_count = store_news_articles(news_articles, st.session_state.user_id)
                    st.info(f"Stored {stored_count} articles in database")
                
                # Analyze news articles with enhanced features
                if news_articles:
                    utils.log_analysis_step("Analyzing news articles")
                    for i, article in enumerate(news_articles):
                        try:
                            # Enhanced analysis with synthetic comparison
                            from analyzer.text_analysis import analyze_with_synthetic_comparison
                            enhanced_analysis = analyze_with_synthetic_comparison(
                                article.get('content', ''),
                                article.get('source', 'Unknown'),
                                topic
                            )
                            
                            analysis = enhanced_analysis['original_analysis']
                            
                            if 'error' not in analysis:
                                # Update news source statistics
                                update_news_source_stats(article.get('source', 'Unknown'), analysis['misinformation_score'])
                                
                                # Store analysis results
                                analysis_id = store_misinfo_analysis(
                                    i+1, 
                                    analysis, 
                                    st.session_state.user_id
                                )
                                
                                # Generate enhanced XAI explanation
                                from analyzer.xai_explanations import generate_comprehensive_explanation_enhanced
                                explanation_data = generate_comprehensive_explanation_enhanced(
                                    article.get('content', ''),
                                    analysis['misinformation_score'],
                                    analysis['confidence_score'],
                                    analysis['llm_origin_probability'],
                                    analysis['trust_score'],
                                    analysis['credibility_score'],
                                    article.get('source', 'Unknown'),
                                    content_type="news",
                                    use_openai=True
                                )
                                
                                article_result = {
                                    'content_type': 'news',
                                    'content_id': article.get('url'),
                                    'title': article.get('title'),
                                    'source': article.get('source'),
                                    'text_analysis': analysis,
                                    'trust_score': analysis['trust_score'],
                                    'credibility_score': analysis['credibility_score'],
                                    'comprehensive_explanation': format_explanation_for_display(explanation_data),
                                    'synthetic_comparison': enhanced_analysis.get('synthetic_samples', [])
                                }
                                results['analysis_results'].append(article_result)
                            else:
                                st.error(f"Analysis error for article {i+1}: {analysis['error']}")
                        except Exception as e:
                            st.error(f"Error analyzing article {i+1}: {e}")
            else:
                st.error("News scraper not available")
        
        # Fetch Reddit posts
        if "Reddit Posts" in content_types:
            utils.log_analysis_step("Fetching Reddit posts", {"topic": topic})
            if fetch_reddit_posts:
                reddit_posts = fetch_reddit_posts(topic, limit=10)
                results['reddit_posts'] = reddit_posts
                
                # Analyze Reddit posts with enhanced features
                if reddit_posts:
                    utils.log_analysis_step("Analyzing Reddit posts")
                    for post in reddit_posts:
                        try:
                            # Enhanced analysis
                            from analyzer.text_analysis import analyze_with_synthetic_comparison
                            enhanced_analysis = analyze_with_synthetic_comparison(
                                post.get('content', ''),
                                'Reddit',
                                topic
                            )
                            
                            analysis = enhanced_analysis['original_analysis']
                            
                            if 'error' not in analysis:
                                # Update subreddit statistics
                                update_subreddit_stats(post.get('subreddit', 'Unknown'), analysis['misinformation_score'])
                                
                                # Generate enhanced XAI explanation
                                from analyzer.xai_explanations import generate_comprehensive_explanation_enhanced
                                explanation_data = generate_comprehensive_explanation_enhanced(
                                    post.get('content', ''),
                                    analysis['misinformation_score'],
                                    analysis['confidence_score'],
                                    analysis['llm_origin_probability'],
                                    analysis['trust_score'],
                                    analysis['credibility_score'],
                                    'Reddit',
                                    content_type="reddit",
                                    use_openai=True
                                )
                                
                                post_result = {
                                    'content_type': 'reddit',
                                    'content_id': post.get('post_id'),
                                    'title': post.get('title'),
                                    'author': post.get('author'),
                                    'subreddit': post.get('subreddit'),
                                    'text_analysis': analysis,
                                    'trust_score': analysis['trust_score'],
                                    'credibility_score': analysis['credibility_score'],
                                    'comprehensive_explanation': format_explanation_for_display(explanation_data),
                                    'synthetic_comparison': enhanced_analysis.get('synthetic_samples', [])
                                }
                                results['analysis_results'].append(post_result)
                            else:
                                st.error(f"Analysis error for Reddit post: {analysis['error']}")
                        except Exception as e:
                            st.error(f"Error analyzing Reddit post: {e}")
            else:
                st.error("Reddit scraper not available")
        
        # Fetch and analyze videos
        if "YouTube Videos" in content_types:
            utils.log_analysis_step("Fetching YouTube videos", {"topic": topic})
            if fetch_videos and download_videos_for_analysis:
                videos = fetch_videos(topic, max_results=5)
                results['videos'] = videos
                
                if videos:
                    # Download videos for analysis
                    utils.log_analysis_step("Downloading videos for analysis")
                    downloaded_videos = download_videos_for_analysis(videos, max_downloads=3)
                    
                    for video in downloaded_videos:
                        try:
                            # Video analysis (lazy import)
                            try:
                                from analyzer.video_analysis import analyze_video
                                video_analysis = analyze_video(video.get('local_video_path'))
                            except Exception as e:
                                print(f"Video analysis error: {e}")
                                video_analysis = {'deepfake_probability': 0.5, 'confidence': 0.0, 'error': str(e)}
                            
                            # Audio analysis (lazy import)
                            try:
                                from analyzer.audio_analysis import analyze_audio
                                audio_analysis = analyze_audio(video.get('local_audio_path'))
                            except Exception as e:
                                print(f"Audio analysis error: {e}")
                                audio_analysis = {'transcript': '', 'misinformation_score': 0.5, 'error': str(e)}
                            
                            # Enhanced analysis for transcript
                            if audio_analysis.get('transcript'):
                                from analyzer.text_analysis import analyze_with_synthetic_comparison
                                enhanced_analysis = analyze_with_synthetic_comparison(
                                    audio_analysis['transcript'],
                                    video.get('channel', 'Unknown'),
                                    topic
                                )
                                transcript_analysis = enhanced_analysis['original_analysis']
                            else:
                                transcript_analysis = {'misinformation_score': 0.5, 'confidence_score': 0.5, 'llm_origin_probability': 0.5}
                            
                            # Update channel statistics
                            update_channel_stats(
                                video.get('channel', 'Unknown'),
                                transcript_analysis.get('misinformation_score', 0.5),
                                video_analysis.get('deepfake_probability', 0.5)
                            )
                            
                            # Generate enhanced explanation
                            from analyzer.xai_explanations import generate_comprehensive_explanation_enhanced
                            explanation_data = generate_comprehensive_explanation_enhanced(
                                audio_analysis.get('transcript', ''),
                                transcript_analysis.get('misinformation_score', 0.5),
                                transcript_analysis.get('confidence_score', 0.5),
                                transcript_analysis.get('llm_origin_probability', 0.5),
                                50.0, 50.0,  # Default trust/credibility
                                video.get('channel', 'Unknown'),
                                content_type="video",
                                use_openai=True
                            )
                            
                            video_result = {
                                'content_type': 'video',
                                'content_id': video.get('video_id'),
                                'title': video.get('title'),
                                'channel': video.get('channel'),
                                'video_analysis': video_analysis,
                                'audio_analysis': audio_analysis,
                                'transcript_analysis': transcript_analysis,
                                'comprehensive_explanation': format_explanation_for_display(explanation_data),
                                'synthetic_comparison': enhanced_analysis.get('synthetic_samples', []) if 'enhanced_analysis' in locals() else []
                            }
                            results['analysis_results'].append(video_result)
                            
                            # Cleanup temporary files
                            if video.get('local_video_path'):
                                utils.cleanup_temp_files([video['local_video_path']])
                            if video.get('local_audio_path'):
                                utils.cleanup_temp_files([video['local_audio_path']])
                                
                        except Exception as e:
                            st.error(f"Error analyzing video {video.get('title', 'Unknown')}: {e}")
            else:
                st.error("Video scraper not available")
        
        # Detect coordinated operations
        if results['analysis_results']:
            utils.log_analysis_step("Detecting coordinated operations")
            from database import detect_coordinated_operations
            coordination_result = detect_coordinated_operations(topic, st.session_state.user_id)
            
            if coordination_result.get('detected'):
                results['coordinated_operations'] = coordination_result
                
                # Generate coordination explanation
                from analyzer.xai_explanations import generate_coordination_explanation
                coordination_explanation = generate_coordination_explanation(coordination_result)
                results['coordinated_operations']['explanation'] = coordination_explanation
        
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        utils.log_analysis_step("Analysis error", {"error": str(e)})
    
    # If no results were found, show error message
    if not results['analysis_results']:
        st.error("❌ No content found from any sources!")
        st.info("""
        **Simplified Analysis Mode:**
        
        The application is now using simplified scrapers that don't require external APIs or complex dependencies.
        
        **For Reddit posts**: Using simulated data for demonstration
        **For YouTube videos**: Using simulated data for demonstration  
        **For news articles**: The news scraper should work automatically
        
        This ensures the application runs without configuration issues.
        """)
    
    return results

def show_results_tab():
    """Show analysis results tab with enhanced features."""
    st.subheader("Analysis Results")
    
    if not st.session_state.analysis_results:
        st.info("No analysis results available. Please run an analysis first.")
        return
    
    results = st.session_state.analysis_results
    
    # Display coordinated operations alert if detected
    if results.get('coordinated_operations') and results['coordinated_operations'].get('detected'):
        st.markdown("---")
        st.markdown("### 🚨 COORDINATED INFLUENCE OPERATION DETECTED")
        
        coord_data = results['coordinated_operations']
        severity = coord_data.get('severity', 'LOW')
        
        if severity == 'HIGH':
            st.error("🔴 **HIGH SEVERITY**: Coordinated influence operation detected across multiple platforms!")
        elif severity == 'MODERATE':
            st.warning("🟡 **MODERATE SEVERITY**: Potential coordinated influence operation detected.")
        else:
            st.info("🔍 **LOW SEVERITY**: Minor coordination patterns detected.")
        
        # Display coordination details
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Platforms Affected", len(coord_data.get('platforms', [])))
            st.metric("Sources Flagged", len(coord_data.get('sources', [])))
        
        with col2:
            st.metric("Coordination Score", f"{coord_data.get('coordination_score', 0):.2f}")
            st.metric("Severity Level", severity)
        
        # Display coordination explanation
        if coord_data.get('explanation'):
            with st.expander("📋 Coordination Analysis Details"):
                st.markdown(coord_data['explanation'])
        
        st.markdown("---")
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(results.get('analysis_results', [])))
    
    with col2:
        avg_misinfo = calculate_average_misinformation(results)
        st.metric("Avg Misinformation Score", f"{avg_misinfo:.1%}")
    
    with col3:
        avg_trust = calculate_average_trust(results)
        st.metric("Avg Trust Score", f"{avg_trust:.1%}")
    
    with col4:
        avg_credibility = calculate_average_credibility(results)
        st.metric("Avg Credibility Score", f"{avg_credibility:.1%}")
    
    # Display detailed results
    st.subheader("Detailed Results")
    
    for i, result in enumerate(results.get('analysis_results', [])):
        with st.expander(f"{result['content_type'].title()}: {result.get('title', 'Untitled')}"):
            display_result_details_enhanced(result)

def display_result_details_enhanced(result):
    """Display enhanced detailed results for a single item."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display comprehensive explanation
        if 'comprehensive_explanation' in result:
            st.markdown("### Analysis Summary")
            st.markdown(result['comprehensive_explanation'])
        
        # Display synthetic comparison if available
        if result.get('synthetic_comparison'):
            with st.expander("🔬 Synthetic Comparison Analysis"):
                st.markdown("**Comparison with AI-generated misinformation samples:**")
                for i, synth in enumerate(result['synthetic_comparison']):
                    st.markdown(f"**Sample {i+1}**: {synth['sample']['title']}")
                    st.markdown(f"Misinfo Score: {synth['analysis'].get('misinformation_score', 0):.1%}")
                    st.markdown(f"Generated by: {synth['sample']['generated_by']}")
                    st.markdown("---")
    
    with col2:
        # Display metrics
        st.markdown("### Key Metrics")
        
        if 'text_analysis' in result:
            text_analysis = result['text_analysis']
            st.metric("Misinformation Score", f"{text_analysis.get('misinformation_score', 0):.1%}")
            st.metric("Confidence Score", f"{text_analysis.get('confidence_score', 0):.1%}")
            st.metric("LLM Origin Probability", f"{text_analysis.get('llm_origin_probability', 0):.1%}")
        
        if 'transcript_analysis' in result:
            transcript_analysis = result['transcript_analysis']
            st.metric("Transcript Misinfo", f"{transcript_analysis.get('misinformation_score', 0):.1%}")
        
        if 'trust_score' in result:
            st.metric("Trust Score", f"{result['trust_score']:.1f}/100")
        
        if 'credibility_score' in result:
            st.metric("Credibility Score", f"{result['credibility_score']:.1f}/100")
        
        if 'video_analysis' in result:
            video_analysis = result['video_analysis']
            st.metric("Deepfake Probability", f"{video_analysis.get('deepfake_probability', 0):.1%}")
    
    # Display alerts based on scores
    display_alerts(result)

def display_alerts(result):
    """Display alerts based on analysis scores."""
    alerts = []
    
    if 'text_analysis' in result:
        misinfo_score = result['text_analysis'].get('misinformation_score', 0)
        if misinfo_score > 0.7:
            alerts.append(("🔴 HIGH MISINFORMATION RISK", "alert-high"))
        elif misinfo_score > 0.4:
            alerts.append(("🟡 MODERATE MISINFORMATION RISK", "alert-medium"))
        else:
            alerts.append(("🟢 LOW MISINFORMATION RISK", "alert-low"))
    
    if 'trust_score' in result:
        trust_score = result['trust_score']
        if trust_score < 30:
            alerts.append(("🔴 LOW TRUST SCORE", "alert-high"))
        elif trust_score < 60:
            alerts.append(("🟡 MODERATE TRUST SCORE", "alert-medium"))
        else:
            alerts.append(("🟢 HIGH TRUST SCORE", "alert-low"))
    
    if 'video_analysis' in result:
        deepfake_prob = result['video_analysis'].get('deepfake_probability', 0)
        if deepfake_prob > 0.7:
            alerts.append(("🔴 HIGH DEEPFAKE RISK", "alert-high"))
        elif deepfake_prob > 0.4:
            alerts.append(("🟡 MODERATE DEEPFAKE RISK", "alert-medium"))
        else:
            alerts.append(("🟢 LOW DEEPFAKE RISK", "alert-low"))
    
    for alert_text, alert_class in alerts:
        st.markdown(f'<div class="{alert_class}">{alert_text}</div>', unsafe_allow_html=True)

def show_history_tab():
    """Show enhanced analysis history tab."""
    st.subheader("Analysis History")
    
    if st.session_state.user_id:
        # Get user's analysis history from database
        history = get_analysis_history(st.session_state.user_id)
        
        if history:
            # Convert to DataFrame for easier display
            history_data = []
            for row in history:
                try:
                    details = json.loads(row[5]) if row[5] else {}
                    history_data.append({
                        'Timestamp': row[7],
                        'Action': row[1],
                        'Content Type': row[2],
                        'Platform': row[9] if len(row) > 9 else 'Unknown',
                        'Source': row[10] if len(row) > 10 else 'Unknown',
                        'Misinfo Score': f"{row[11]:.2f}" if len(row) > 11 and row[11] else 'N/A',
                        'Success': '✅' if row[8] else '❌'
                    })
                except:
                    continue
            
            if history_data:
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No detailed history available.")
        else:
            st.info("No analysis history found.")
        
        # Show coordinated operations history
        from database import get_coordinated_operations
        coord_ops = get_coordinated_operations(st.session_state.user_id)
        
        if coord_ops:
            st.subheader("Coordinated Operations History")
            coord_data = []
            for op in coord_ops:
                try:
                    platforms = json.loads(op[3]) if op[3] else []
                    sources = json.loads(op[2]) if op[2] else []
                    coord_data.append({
                        'Date': op[8],
                        'Topic': op[1],
                        'Platforms': ', '.join(platforms),
                        'Sources': len(sources),
                        'Severity': op[6],
                        'Score': f"{op[5]:.2f}"
                    })
                except:
                    continue
            
            if coord_data:
                coord_df = pd.DataFrame(coord_data)
                st.dataframe(coord_df, use_container_width=True)
    else:
        st.info("Please login to view analysis history.")

def show_about_tab():
    """Show enhanced about tab."""
    st.subheader("About VERITAS.AI")
    
    st.markdown("""
    ### Mission
    VERITAS.AI is dedicated to combating misinformation and promoting digital literacy through advanced AI-powered content analysis.
    
    ### Enhanced Features
    - **Multi-Modal Analysis**: Analyze text, audio, and video content
    - **Synthetic Misinformation Generation**: Compare real content with AI-generated samples
    - **Coordinated Operations Detection**: Identify influence campaigns across platforms
    - **Enhanced LLM Origin Detection**: Multi-model ensemble for AI-generated content detection
    - **OpenAI Integration**: Expert AI explanations for suspicious content
    - **Source Reputation Tracking**: Comprehensive credibility scoring
    - **Real-time Flagging**: Automatic flagging of unreliable sources and channels
    
    ### Technology Stack
    - **Text Analysis**: Transformers-based models (mrm8488/bert-tiny-fake-news-detection)
    - **LLM Detection**: Ensemble models (DialoGPT, GPT-2 detector, custom classifiers)
    - **Audio Analysis**: OpenAI Whisper for transcription and analysis
    - **Video Analysis**: Computer vision for deepfake detection
    - **Synthetic Generation**: GPT-2 and Mistral models for comparison
    - **Trust Assessment**: Enhanced heuristic-based credibility scoring
    - **Web Interface**: Streamlit for user-friendly interaction
    
    ### Privacy & Security
    - All analysis is performed locally when possible
    - User data is stored securely in local database
    - No content is shared with third parties (except OpenAI API when enabled)
    - Temporary files are automatically cleaned up
    
    ### Getting Started
    1. Register or login to your account
    2. Enter a topic you want to analyze
    3. Select content types (news, Reddit, videos)
    4. Review comprehensive analysis results with AI explanations
    5. Check for coordinated influence operations
    6. Monitor flagged sources and channels
    """)
    
    # API Configuration Status
    st.subheader("API Configuration Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check Reddit Analysis
        st.success("✅ Reddit Analysis Available")
        st.info("Using simplified Reddit analysis without API")
    
    with col2:
        # Check OpenAI API
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if openai_api_key:
            st.success("✅ OpenAI API Configured")
        else:
            st.warning("⚠️ OpenAI API Not Configured")
            st.info("Add OPENAI_API_KEY to .env file for AI explanations")
    
    with col3:
        # Check Video Analysis
        st.success("✅ Video Analysis Available")
        st.info("Using simplified video analysis without FFmpeg")

def calculate_average_misinformation(results):
    """Calculate average misinformation score."""
    scores = []
    for result in results.get('analysis_results', []):
        if 'text_analysis' in result:
            scores.append(result['text_analysis'].get('misinformation_score', 0))
    return sum(scores) / len(scores) if scores else 0

def calculate_average_trust(results):
    """Calculate average trust score."""
    scores = []
    for result in results.get('analysis_results', []):
        if 'trust_score' in result:
            scores.append(result['trust_score'])
    return sum(scores) / len(scores) if scores else 0

def calculate_average_credibility(results):
    """Calculate average credibility score."""
    scores = []
    for result in results.get('analysis_results', []):
        if 'credibility_score' in result:
            scores.append(result['credibility_score'])
    return sum(scores) / len(scores) if scores else 0

if __name__ == "__main__":
    main() 