"""
VERITAS.AI - Comprehensive Misinformation Detection Pipeline

This module provides the main pipeline for analyzing content across multiple platforms
and detecting misinformation, LLM-generated content, and coordinated influence operations.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from .text_analysis import (
    comprehensive_text_analysis,
    analyze_with_synthetic_comparison,
    assess_risk_level
)

from .xai_explanations import (
    generate_comprehensive_explanation_enhanced,
    format_explanation_for_display_enhanced
)

from .trust_credibility import (
    calculate_trust_score,
    calculate_credibility_score,
    generate_trust_explanation
)

try:
    from config import (
        MISINFO_THRESHOLD,
        LLM_DETECTION_THRESHOLD,
        SUBREDDIT_FLAG_THRESHOLD,
        CHANNEL_FLAG_THRESHOLD,
        REDDIT_ANALYSIS_CONFIG,
        VIDEO_ANALYSIS_CONFIG
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        MISINFO_THRESHOLD,
        LLM_DETECTION_THRESHOLD,
        SUBREDDIT_FLAG_THRESHOLD,
        CHANNEL_FLAG_THRESHOLD,
        REDDIT_ANALYSIS_CONFIG,
        VIDEO_ANALYSIS_CONFIG
    )

class MisinformationDetectionPipeline:
    """Main pipeline for comprehensive misinformation detection."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.analysis_results = {}
        self.flagged_sources = {}
        self.coordinated_operations = []
        self.analysis_timestamp = datetime.now()
    
    def analyze_reddit_content(self, reddit_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Analyze Reddit content comprehensively."""
        try:
            print("🔍 Starting Reddit content analysis...")
            
            posts = reddit_data.get('posts', [])
            comments = reddit_data.get('comments', [])
            subreddit_metadata = reddit_data.get('subreddit_metadata', {})
            
            # Analyze posts
            post_analyses = []
            subreddit_misinfo_counts = {}
            
            for post in posts:
                print(f"Analyzing post: {post.get('title', 'Unknown')[:50]}...")
                
                # Combine title and content for analysis
                content = f"{post.get('title', '')} {post.get('content', '')}"
                
                # Perform comprehensive analysis
                analysis = comprehensive_text_analysis(
                    content, 
                    source=post.get('subreddit', 'Unknown'),
                    content_type="reddit"
                )
                
                # Add post-specific metadata
                analysis['post_metadata'] = {
                    'post_id': post.get('post_id'),
                    'author': post.get('author'),
                    'subreddit': post.get('subreddit'),
                    'score': post.get('score'),
                    'upvote_ratio': post.get('upvote_ratio'),
                    'num_comments': post.get('num_comments'),
                    'created_utc': post.get('created_utc'),
                    'url': post.get('url')
                }
                
                # Track subreddit misinformation counts
                subreddit = post.get('subreddit', 'Unknown')
                if subreddit not in subreddit_misinfo_counts:
                    subreddit_misinfo_counts[subreddit] = 0
                
                if analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                    subreddit_misinfo_counts[subreddit] += 1
                
                post_analyses.append(analysis)
            
            # Analyze comments
            comment_analyses = []
            for comment in comments:
                print(f"Analyzing comment from {comment.get('author', 'Unknown')}...")
                
                analysis = comprehensive_text_analysis(
                    comment.get('content', ''),
                    source=comment.get('subreddit', 'Unknown'),
                    content_type="reddit"
                )
                
                analysis['comment_metadata'] = {
                    'comment_id': comment.get('comment_id'),
                    'author': comment.get('author'),
                    'subreddit': comment.get('subreddit'),
                    'score': comment.get('score'),
                    'parent_post_id': comment.get('parent_post_id'),
                    'created_utc': comment.get('created_utc')
                }
                
                comment_analyses.append(analysis)
            
            # Check for subreddit flagging
            flagged_subreddits = []
            for subreddit, count in subreddit_misinfo_counts.items():
                if count >= SUBREDDIT_FLAG_THRESHOLD:
                    flagged_subreddits.append({
                        'subreddit': subreddit,
                        'misinfo_post_count': count,
                        'total_posts': len([p for p in posts if p.get('subreddit') == subreddit]),
                        'flag_reason': f"Multiple posts ({count}) with high misinformation scores"
                    })
            
            # Generate explanations
            post_explanations = []
            for analysis in post_analyses:
                explanation = generate_comprehensive_explanation_enhanced(
                    content=analysis.get('post_metadata', {}).get('title', '') + ' ' + analysis.get('post_metadata', {}).get('content', ''),
                    misinfo_score=analysis.get('misinformation_score', 0),
                    confidence_score=analysis.get('confidence_score', 0),
                    llm_prob=analysis.get('llm_origin_probability', 0),
                    trust_score=analysis.get('trust_score', 0),
                    credibility_score=analysis.get('credibility_score', 0),
                    source=analysis.get('source', 'Unknown'),
                    content_type="reddit"
                )
                post_explanations.append(explanation)
            
            # Detect suspicious patterns
            suspicious_patterns = self._detect_reddit_patterns(posts, comments)
            
            return {
                'posts_analyzed': len(posts),
                'comments_analyzed': len(comments),
                'post_analyses': post_analyses,
                'comment_analyses': comment_analyses,
                'post_explanations': post_explanations,
                'flagged_subreddits': flagged_subreddits,
                'suspicious_patterns': suspicious_patterns,
                'subreddit_metadata': subreddit_metadata,
                'analysis_timestamp': datetime.now().isoformat(),
                'platform': 'reddit'
            }
            
        except Exception as e:
            print(f"Error in Reddit analysis: {e}")
            return {
                'error': str(e),
                'posts_analyzed': 0,
                'comments_analyzed': 0,
                'platform': 'reddit'
            }
    
    def analyze_youtube_content(self, video_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Analyze YouTube content comprehensively."""
        try:
            print("📹 Starting YouTube content analysis...")
            
            videos = video_data.get('videos', [])
            downloaded_videos = video_data.get('downloaded_videos', [])
            processed_videos = video_data.get('processed_videos', [])
            
            # Analyze video metadata
            video_analyses = []
            channel_misinfo_counts = {}
            
            for video in videos:
                print(f"Analyzing video: {video.get('title', 'Unknown')[:50]}...")
                
                # Analyze title and description
                content = f"{video.get('title', '')} {video.get('description', '')}"
                
                analysis = comprehensive_text_analysis(
                    content,
                    source=video.get('channel', 'Unknown'),
                    content_type="video"
                )
                
                # Add video-specific metadata
                analysis['video_metadata'] = {
                    'video_id': video.get('video_id'),
                    'title': video.get('title'),
                    'channel': video.get('channel'),
                    'channel_id': video.get('channel_id'),
                    'duration': video.get('duration'),
                    'view_count': video.get('view_count'),
                    'like_count': video.get('like_count'),
                    'comment_count': video.get('comment_count'),
                    'published_at': video.get('published_at'),
                    'url': video.get('url')
                }
                
                # Track channel misinformation counts
                channel = video.get('channel', 'Unknown')
                if channel not in channel_misinfo_counts:
                    channel_misinfo_counts[channel] = 0
                
                if analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                    channel_misinfo_counts[channel] += 1
                
                video_analyses.append(analysis)
            
            # Analyze processed videos (with transcripts and frame analysis)
            processed_analyses = []
            for processed_video in processed_videos:
                print(f"Analyzing processed video: {processed_video.get('title', 'Unknown')[:50]}...")
                
                transcript_data = processed_video.get('transcript', {})
                frame_analysis = processed_video.get('frame_analysis', {})
                
                # Analyze transcript
                transcript_text = transcript_data.get('text', '')
                if transcript_text:
                    transcript_analysis = comprehensive_text_analysis(
                        transcript_text,
                        source=processed_video.get('channel', 'Unknown'),
                        content_type="video"
                    )
                else:
                    transcript_analysis = {
                        'misinformation_score': 0.0,
                        'confidence_score': 0.0,
                        'llm_origin_probability': 0.0,
                        'trust_score': 50.0,
                        'credibility_score': 50.0,
                        'risk_level': 'UNKNOWN'
                    }
                
                # Combine transcript and frame analysis
                combined_analysis = {
                    **transcript_analysis,
                    'video_id': processed_video.get('video_id'),
                    'title': processed_video.get('title'),
                    'channel': processed_video.get('channel'),
                    'transcript_analysis': transcript_analysis,
                    'frame_analysis': frame_analysis,
                    'deepfake_score': frame_analysis.get('deepfake_score', 0.0),
                    'temporal_consistency': frame_analysis.get('temporal_consistency', 0.0),
                    'compression_artifacts': frame_analysis.get('compression_artifacts', 0.0),
                    'transcript_confidence': transcript_data.get('transcription_confidence', 0.0),
                    'language': transcript_data.get('language', 'unknown'),
                    'processing_timestamp': processed_video.get('processing_timestamp')
                }
                
                processed_analyses.append(combined_analysis)
            
            # Check for channel flagging
            flagged_channels = []
            for channel, count in channel_misinfo_counts.items():
                if count >= CHANNEL_FLAG_THRESHOLD:
                    flagged_channels.append({
                        'channel': channel,
                        'misinfo_video_count': count,
                        'total_videos': len([v for v in videos if v.get('channel') == channel]),
                        'flag_reason': f"Multiple videos ({count}) with high misinformation scores"
                    })
            
            # Generate explanations
            video_explanations = []
            for analysis in video_analyses:
                content = analysis.get('video_metadata', {}).get('title', '') + ' ' + analysis.get('video_metadata', {}).get('description', '')
                explanation = generate_comprehensive_explanation_enhanced(
                    content=content,
                    misinfo_score=analysis.get('misinformation_score', 0),
                    confidence_score=analysis.get('confidence_score', 0),
                    llm_prob=analysis.get('llm_origin_probability', 0),
                    trust_score=analysis.get('trust_score', 0),
                    credibility_score=analysis.get('credibility_score', 0),
                    source=analysis.get('source', 'Unknown'),
                    content_type="video"
                )
                video_explanations.append(explanation)
            
            return {
                'videos_analyzed': len(videos),
                'videos_downloaded': len(downloaded_videos),
                'videos_processed': len(processed_videos),
                'video_analyses': video_analyses,
                'processed_analyses': processed_analyses,
                'video_explanations': video_explanations,
                'flagged_channels': flagged_channels,
                'analysis_timestamp': datetime.now().isoformat(),
                'platform': 'youtube'
            }
            
        except Exception as e:
            print(f"Error in YouTube analysis: {e}")
            return {
                'error': str(e),
                'videos_analyzed': 0,
                'videos_downloaded': 0,
                'videos_processed': 0,
                'platform': 'youtube'
            }
    
    def analyze_news_content(self, news_data: List[Dict[str, Any]], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Analyze news content comprehensively."""
        try:
            print("📰 Starting news content analysis...")
            
            news_analyses = []
            source_misinfo_counts = {}
            
            for article in news_data:
                print(f"Analyzing article: {article.get('title', 'Unknown')[:50]}...")
                
                content = f"{article.get('title', '')} {article.get('content', '')}"
                
                analysis = comprehensive_text_analysis(
                    content,
                    source=article.get('source', 'Unknown'),
                    content_type="news"
                )
                
                # Add article-specific metadata
                analysis['article_metadata'] = {
                    'url': article.get('url'),
                    'title': article.get('title'),
                    'source': article.get('source'),
                    'published_date': article.get('published_date'),
                    'authors': article.get('authors'),
                    'keywords': article.get('keywords')
                }
                
                # Track source misinformation counts
                source = article.get('source', 'Unknown')
                if source not in source_misinfo_counts:
                    source_misinfo_counts[source] = 0
                
                if analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                    source_misinfo_counts[source] += 1
                
                news_analyses.append(analysis)
            
            # Check for source flagging
            flagged_sources = []
            for source, count in source_misinfo_counts.items():
                if count >= 3:  # Flag sources with 3+ high-misinfo articles
                    flagged_sources.append({
                        'source': source,
                        'misinfo_article_count': count,
                        'total_articles': len([a for a in news_data if a.get('source') == source]),
                        'flag_reason': f"Multiple articles ({count}) with high misinformation scores"
                    })
            
            # Generate explanations
            news_explanations = []
            for analysis in news_analyses:
                content = analysis.get('article_metadata', {}).get('title', '') + ' ' + analysis.get('article_metadata', {}).get('content', '')
                explanation = generate_comprehensive_explanation_enhanced(
                    content=content,
                    misinfo_score=analysis.get('misinformation_score', 0),
                    confidence_score=analysis.get('confidence_score', 0),
                    llm_prob=analysis.get('llm_origin_probability', 0),
                    trust_score=analysis.get('trust_score', 0),
                    credibility_score=analysis.get('credibility_score', 0),
                    source=analysis.get('source', 'Unknown'),
                    content_type="news"
                )
                news_explanations.append(explanation)
            
            return {
                'articles_analyzed': len(news_data),
                'news_analyses': news_analyses,
                'news_explanations': news_explanations,
                'flagged_sources': flagged_sources,
                'analysis_timestamp': datetime.now().isoformat(),
                'platform': 'news'
            }
            
        except Exception as e:
            print(f"Error in news analysis: {e}")
            return {
                'error': str(e),
                'articles_analyzed': 0,
                'platform': 'news'
            }
    
    def detect_coordinated_operations(self, all_analyses: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Detect coordinated influence operations across platforms."""
        try:
            print("🔍 Detecting coordinated operations...")
            
            # Collect all flagged content
            flagged_content = []
            
            # From Reddit analysis
            reddit_analysis = all_analyses.get('reddit', {})
            for post_analysis in reddit_analysis.get('post_analyses', []):
                if post_analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                    flagged_content.append({
                        'platform': 'reddit',
                        'content_type': 'post',
                        'source': post_analysis.get('source'),
                        'author': post_analysis.get('post_metadata', {}).get('author'),
                        'misinfo_score': post_analysis.get('misinformation_score'),
                        'llm_prob': post_analysis.get('llm_origin_probability'),
                        'timestamp': post_analysis.get('post_metadata', {}).get('created_utc')
                    })
            
            # From YouTube analysis
            youtube_analysis = all_analyses.get('youtube', {})
            for video_analysis in youtube_analysis.get('video_analyses', []):
                if video_analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                    flagged_content.append({
                        'platform': 'youtube',
                        'content_type': 'video',
                        'source': video_analysis.get('source'),
                        'author': video_analysis.get('video_metadata', {}).get('channel'),
                        'misinfo_score': video_analysis.get('misinformation_score'),
                        'llm_prob': video_analysis.get('llm_origin_probability'),
                        'timestamp': video_analysis.get('video_metadata', {}).get('published_at')
                    })
            
            # From news analysis
            news_analysis = all_analyses.get('news', {})
            for article_analysis in news_analysis.get('news_analyses', []):
                if article_analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                    flagged_content.append({
                        'platform': 'news',
                        'content_type': 'article',
                        'source': article_analysis.get('source'),
                        'author': article_analysis.get('article_metadata', {}).get('authors'),
                        'misinfo_score': article_analysis.get('misinformation_score'),
                        'llm_prob': article_analysis.get('llm_origin_probability'),
                        'timestamp': article_analysis.get('article_metadata', {}).get('published_date')
                    })
            
            # Analyze for coordination patterns
            coordination_analysis = self._analyze_coordination_patterns(flagged_content, topic)
            
            return {
                'flagged_content_count': len(flagged_content),
                'flagged_content': flagged_content,
                'coordination_analysis': coordination_analysis,
                'detection_timestamp': datetime.now().isoformat(),
                'topic': topic
            }
            
        except Exception as e:
            print(f"Error in coordination detection: {e}")
            return {
                'error': str(e),
                'flagged_content_count': 0,
                'topic': topic
            }
    
    def _detect_reddit_patterns(self, posts: List[Dict], comments: List[Dict]) -> Dict[str, Any]:
        """Detect suspicious patterns in Reddit content."""
        try:
            patterns = {
                'coordinated_posting': False,
                'bot_indicators': [],
                'spam_indicators': [],
                'manipulation_indicators': []
            }
            
            # Check for coordinated posting (same author, similar content, same time)
            authors = {}
            for post in posts:
                author = post.get('author', '')
                if author not in authors:
                    authors[author] = []
                authors[author].append(post)
            
            # Flag authors with multiple posts in short time
            for author, author_posts in authors.items():
                if len(author_posts) > 3:
                    patterns['coordinated_posting'] = True
                    patterns['bot_indicators'].append(f"Author {author} posted {len(author_posts)} times")
            
            # Check for spam indicators
            spam_keywords = ['free', 'money', 'click', 'buy', 'limited', 'offer', 'discount']
            for post in posts:
                content = (post.get('title', '') + ' ' + post.get('content', '')).lower()
                spam_count = sum(1 for keyword in spam_keywords if keyword in content)
                if spam_count > 2:
                    patterns['spam_indicators'].append(f"Post {post.get('post_id')} contains {spam_count} spam keywords")
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting Reddit patterns: {e}")
            return {}
    
    def _analyze_coordination_patterns(self, flagged_content: List[Dict], topic: str) -> Dict[str, Any]:
        """Analyze patterns for coordinated influence operations."""
        try:
            analysis = {
                'coordination_score': 0.0,
                'severity_level': 'LOW',
                'patterns_detected': [],
                'affected_platforms': set(),
                'affected_sources': set(),
                'time_clustering': False,
                'content_similarity': False
            }
            
            if not flagged_content:
                return analysis
            
            # Collect platforms and sources
            for content in flagged_content:
                analysis['affected_platforms'].add(content.get('platform', 'unknown'))
                analysis['affected_sources'].add(content.get('source', 'unknown'))
            
            # Convert sets to lists for JSON serialization
            analysis['affected_platforms'] = list(analysis['affected_platforms'])
            analysis['affected_sources'] = list(analysis['affected_sources'])
            
            # Calculate coordination score based on various factors
            platform_diversity = len(analysis['affected_platforms'])
            source_diversity = len(analysis['affected_sources'])
            content_count = len(flagged_content)
            
            # Higher scores for more diverse coordination
            coordination_score = (platform_diversity * 0.3 + source_diversity * 0.2 + content_count * 0.1)
            analysis['coordination_score'] = min(1.0, coordination_score)
            
            # Determine severity level
            if analysis['coordination_score'] > 0.8:
                analysis['severity_level'] = 'HIGH'
            elif analysis['coordination_score'] > 0.5:
                analysis['severity_level'] = 'MODERATE'
            else:
                analysis['severity_level'] = 'LOW'
            
            # Detect specific patterns
            if platform_diversity > 2:
                analysis['patterns_detected'].append(f"Cross-platform coordination across {platform_diversity} platforms")
            
            if source_diversity > 5:
                analysis['patterns_detected'].append(f"Multiple sources ({source_diversity}) involved")
            
            if content_count > 10:
                analysis['patterns_detected'].append(f"High volume of flagged content ({content_count} items)")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing coordination patterns: {e}")
            return {
                'coordination_score': 0.0,
                'severity_level': 'UNKNOWN',
                'patterns_detected': [],
                'affected_platforms': [],
                'affected_sources': [],
                'error': str(e)
            }
    
    def generate_comprehensive_report(self, all_analyses: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        try:
            print("📊 Generating comprehensive report...")
            
            # Collect summary statistics
            total_content_analyzed = 0
            total_flagged_content = 0
            avg_misinfo_score = 0.0
            avg_llm_prob = 0.0
            
            all_scores = []
            
            # Process each platform's analysis
            for platform, analysis in all_analyses.items():
                if platform == 'reddit':
                    total_content_analyzed += analysis.get('posts_analyzed', 0) + analysis.get('comments_analyzed', 0)
                    for post_analysis in analysis.get('post_analyses', []):
                        all_scores.append(post_analysis.get('misinformation_score', 0))
                        if post_analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                            total_flagged_content += 1
                
                elif platform == 'youtube':
                    total_content_analyzed += analysis.get('videos_analyzed', 0)
                    for video_analysis in analysis.get('video_analyses', []):
                        all_scores.append(video_analysis.get('misinformation_score', 0))
                        if video_analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                            total_flagged_content += 1
                
                elif platform == 'news':
                    total_content_analyzed += analysis.get('articles_analyzed', 0)
                    for article_analysis in analysis.get('news_analyses', []):
                        all_scores.append(article_analysis.get('misinformation_score', 0))
                        if article_analysis.get('misinformation_score', 0) > MISINFO_THRESHOLD:
                            total_flagged_content += 1
            
            # Calculate averages
            if all_scores:
                avg_misinfo_score = sum(all_scores) / len(all_scores)
            
            # Generate risk assessment
            risk_level = 'LOW'
            if avg_misinfo_score > 0.7 or total_flagged_content > 10:
                risk_level = 'HIGH'
            elif avg_misinfo_score > 0.5 or total_flagged_content > 5:
                risk_level = 'MODERATE'
            
            # Create comprehensive report
            report = {
                'topic': topic,
                'analysis_timestamp': datetime.now().isoformat(),
                'summary_statistics': {
                    'total_content_analyzed': total_content_analyzed,
                    'total_flagged_content': total_flagged_content,
                    'flagging_rate': total_flagged_content / max(1, total_content_analyzed),
                    'average_misinformation_score': avg_misinfo_score,
                    'average_llm_probability': avg_llm_prob,
                    'overall_risk_level': risk_level
                },
                'platform_analyses': all_analyses,
                'flagged_sources': {
                    'subreddits': all_analyses.get('reddit', {}).get('flagged_subreddits', []),
                    'channels': all_analyses.get('youtube', {}).get('flagged_channels', []),
                    'news_sources': all_analyses.get('news', {}).get('flagged_sources', [])
                },
                'coordination_analysis': all_analyses.get('coordination', {}),
                'recommendations': self._generate_recommendations(all_analyses, risk_level)
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            return {
                'error': str(e),
                'topic': topic,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, all_analyses: Dict[str, Any], risk_level: str) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        try:
            if risk_level == 'HIGH':
                recommendations.extend([
                    "🚨 HIGH RISK DETECTED - Exercise extreme caution",
                    "🔍 Verify all claims through multiple reliable sources",
                    "📰 Check fact-checking websites for this topic",
                    "⚠️ Avoid sharing content without thorough verification",
                    "👥 Be aware of potential coordinated influence operations"
                ])
            elif risk_level == 'MODERATE':
                recommendations.extend([
                    "⚠️ MODERATE RISK DETECTED - Approach with skepticism",
                    "🔍 Cross-reference information with established sources",
                    "📊 Look for multiple perspectives on this topic",
                    "🤔 Consider the source's track record and credibility"
                ])
            else:
                recommendations.extend([
                    "✅ LOW RISK DETECTED - Content appears relatively reliable",
                    "🔍 Still verify important claims independently",
                    "📚 Consult multiple sources for comprehensive understanding"
                ])
            
            # Platform-specific recommendations
            if 'reddit' in all_analyses:
                flagged_subreddits = all_analyses['reddit'].get('flagged_subreddits', [])
                if flagged_subreddits:
                    recommendations.append(f"📱 Exercise caution with {len(flagged_subreddits)} flagged subreddits")
            
            if 'youtube' in all_analyses:
                flagged_channels = all_analyses['youtube'].get('flagged_channels', [])
                if flagged_channels:
                    recommendations.append(f"📺 Exercise caution with {len(flagged_channels)} flagged YouTube channels")
            
            if 'news' in all_analyses:
                flagged_sources = all_analyses['news'].get('flagged_sources', [])
                if flagged_sources:
                    recommendations.append(f"📰 Exercise caution with {len(flagged_sources)} flagged news sources")
            
            # Always include general recommendations
            recommendations.extend([
                "📚 Consult multiple reliable news sources",
                "🎯 Focus on facts over sensationalist language",
                "⏰ Be patient - wait for verified information",
                "🔄 Stay updated with fact-checking organizations"
            ])
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Unable to generate specific recommendations due to technical issues"]

# Convenience function for running the complete pipeline
def run_comprehensive_analysis(
    topic: str,
    reddit_data: Optional[Dict[str, Any]] = None,
    youtube_data: Optional[Dict[str, Any]] = None,
    news_data: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Run comprehensive misinformation detection analysis across all platforms."""
    
    pipeline = MisinformationDetectionPipeline()
    all_analyses = {}
    
    try:
        # Analyze Reddit content
        if reddit_data:
            print("🔍 Analyzing Reddit content...")
            all_analyses['reddit'] = pipeline.analyze_reddit_content(reddit_data, user_id)
        
        # Analyze YouTube content
        if youtube_data:
            print("📹 Analyzing YouTube content...")
            all_analyses['youtube'] = pipeline.analyze_youtube_content(youtube_data, user_id)
        
        # Analyze news content
        if news_data:
            print("📰 Analyzing news content...")
            all_analyses['news'] = pipeline.analyze_news_content(news_data, user_id)
        
        # Detect coordinated operations
        print("🔍 Detecting coordinated operations...")
        all_analyses['coordination'] = pipeline.detect_coordinated_operations(all_analyses, topic)
        
        # Generate comprehensive report
        print("📊 Generating comprehensive report...")
        report = pipeline.generate_comprehensive_report(all_analyses, topic)
        
        return report
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        return {
            'error': str(e),
            'topic': topic,
            'analysis_timestamp': datetime.now().isoformat()
        } 