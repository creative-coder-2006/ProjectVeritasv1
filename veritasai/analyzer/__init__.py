"""
VERITAS.AI - Comprehensive Misinformation Detection Pipeline

This module provides comprehensive analysis capabilities for detecting misinformation
across multiple platforms including news articles, Reddit posts/comments, and YouTube videos.

Features:
- Text-based misinformation detection using BERT models
- LLM origin detection using statistical and model-based approaches
- Video analysis with audio transcription and frame analysis
- Reddit-specific analysis with subreddit flagging
- YouTube-specific analysis with channel flagging
- Explainable AI with OpenAI integration
- Trust and credibility scoring
- Coordinated influence operation detection
"""

from .text_analysis import (
    analyze_text,
    analyze_multiple_texts,
    comprehensive_text_analysis,
    detect_llm_origin,
    calculate_trust_score,
    calculate_credibility_score,
    generate_synthetic_misinformation,
    analyze_with_synthetic_comparison,
    assess_risk_level,
    analyze_feature_importance
)

from .video_analysis import (
    VideoAnalyzer,
    analyze_video,
    analyze_multiple_videos
)

from .audio_analysis import (
    transcribe_audio,
    analyze_audio,
    analyze_multiple_audio_files
)

from .trust_credibility import (
    calculate_trust_score,
    calculate_credibility_score,
    generate_trust_explanation
)

from .xai_explanations import (
    generate_comprehensive_explanation,
    generate_reddit_explanation,
    generate_youtube_explanation,
    generate_openai_style_explanation,
    format_explanation_for_display,
    generate_coordination_explanation,
    generate_comprehensive_explanation_enhanced,
    format_explanation_for_display_enhanced
)

__all__ = [
    # Text Analysis
    'analyze_text',
    'analyze_multiple_texts',
    'comprehensive_text_analysis',
    'detect_llm_origin',
    'calculate_trust_score',
    'calculate_credibility_score',
    'generate_synthetic_misinformation',
    'analyze_with_synthetic_comparison',
    'assess_risk_level',
    'analyze_feature_importance',
    
    # Video Analysis
    'VideoAnalyzer',
    'analyze_video',
    'analyze_multiple_videos',
    
    # Audio Analysis
    'transcribe_audio',
    'analyze_audio',
    'analyze_multiple_audio_files',
    
    # Trust & Credibility
    'calculate_trust_score',
    'calculate_credibility_score',
    'generate_trust_explanation',
    
    # XAI Explanations
    'generate_comprehensive_explanation',
    'generate_reddit_explanation',
    'generate_youtube_explanation',
    'generate_openai_style_explanation',
    'format_explanation_for_display',
    'generate_coordination_explanation',
    'generate_comprehensive_explanation_enhanced',
    'format_explanation_for_display_enhanced'
] 