import json
import numpy as np
from typing import Dict, Any, List, Optional
import streamlit as st
import openai
try:
    from config import OPENAI_API_KEY
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import OPENAI_API_KEY
from datetime import datetime

def generate_comprehensive_explanation(
    article_content: str,
    misinfo_score: float,
    confidence_score: float,
    llm_prob: float,
    trust_score: float,
    credibility_score: float,
    source: str = "Unknown",
    content_type: str = "news",
    platform: str = "unknown"
) -> Dict[str, Any]:
    """Generate comprehensive XAI explanation for analysis results."""
    
    try:
        # Determine explanation type based on scores
        if misinfo_score > 0.7:
            explanation_type = "high_misinformation"
        elif misinfo_score > 0.5:
            explanation_type = "moderate_misinformation"
        elif llm_prob > 0.7:
            explanation_type = "likely_ai_generated"
        else:
            explanation_type = "likely_credible"
        
        # Generate explanation based on type and platform
        if content_type == "reddit":
            explanation = generate_reddit_explanation(
                article_content, misinfo_score, confidence_score, llm_prob, source
            )
        elif content_type == "video":
            explanation = generate_youtube_explanation(
                article_content, misinfo_score, confidence_score, llm_prob, source
            )
        else:
            explanation = generate_news_explanation(
                article_content, misinfo_score, confidence_score, llm_prob, source
            )
        
        # Generate feature importance
        feature_importance = analyze_feature_importance(article_content)
        
        # Generate AI explanation for suspicious content
        ai_explanation = ""
        if misinfo_score > 0.5 or llm_prob > 0.7:
            ai_explanation = generate_openai_style_explanation(
                article_content, misinfo_score, confidence_score, llm_prob, content_type, platform
            )
        
        return {
            'explanation_type': explanation_type,
            'explanation_text': explanation,
            'feature_importance': feature_importance,
            'ai_explanation': ai_explanation,
            'risk_level': get_risk_level(misinfo_score, llm_prob),
            'recommendations': generate_recommendations(misinfo_score, llm_prob, source, content_type),
            'platform': platform,
            'content_type': content_type
        }
        
    except Exception as e:
        print(f"Error generating comprehensive explanation: {e}")
        return {
            'explanation_type': 'error',
            'explanation_text': f"Error generating explanation: {str(e)}",
            'feature_importance': {},
            'ai_explanation': "",
            'risk_level': 'unknown',
            'recommendations': ['Unable to generate recommendations due to error'],
            'platform': platform,
            'content_type': content_type
        }

def generate_reddit_explanation(
    content: str, 
    misinfo_score: float, 
    confidence_score: float, 
    llm_prob: float, 
    source: str
) -> str:
    """Generate explanation for Reddit content using the master prompt."""
    
    explanation = f"""
    **🔍 REDDIT CONTENT ANALYSIS**
    
    **Post/Comment Analysis Results:**
    • Misinformation Score: {misinfo_score:.1%}
    • Confidence Level: {confidence_score:.1%}
    • LLM Origin Probability: {llm_prob:.1%}
    • Subreddit: {source}
    
    **Analysis Breakdown:**
    """
    
    if misinfo_score > 0.7:
        explanation += f"""
        🚨 **HIGH MISINFORMATION RISK DETECTED**
        
        This Reddit content shows strong indicators of potential misinformation:
        • Multiple red flags detected in the analysis
        • Content patterns suggest unreliable information
        • Subreddit context may contribute to risk level
        
        **Key Concerns:**
        • Misinformation probability: {misinfo_score:.1%}
        • Confidence in assessment: {confidence_score:.1%}
        • Potential AI-generated content: {llm_prob:.1%}
        
        **Recommendation:** Exercise extreme caution. Verify claims through multiple reliable sources before sharing or believing this content.
        """
    elif misinfo_score > 0.5:
        explanation += f"""
        ⚠️ **MODERATE MISINFORMATION RISK**
        
        This Reddit content shows some concerning indicators:
        • Moderate risk of misinformation detected
        • Some red flags present in the analysis
        • Mixed reliability indicators
        
        **Assessment Details:**
        • Misinformation probability: {misinfo_score:.1%}
        • Confidence level: {confidence_score:.1%}
        • AI generation likelihood: {llm_prob:.1%}
        
        **Recommendation:** Approach with skepticism. Cross-reference information with established sources.
        """
    elif llm_prob > 0.7:
        explanation += f"""
        🤖 **LIKELY AI-GENERATED CONTENT**
        
        This Reddit content appears to be generated by an AI system:
        • High probability of LLM origin: {llm_prob:.1%}
        • Content patterns suggest automated generation
        • May be part of coordinated influence operations
        
        **Analysis:**
        • Misinformation score: {misinfo_score:.1%}
        • Confidence: {confidence_score:.1%}
        • AI generation probability: {llm_prob:.1%}
        
        **Recommendation:** Be aware this content may be artificially generated. Verify authenticity and sources.
        """
    else:
        explanation += f"""
        ✅ **LIKELY CREDIBLE CONTENT**
        
        This Reddit content appears to be relatively reliable:
        • Low misinformation risk: {misinfo_score:.1%}
        • Low AI generation probability: {llm_prob:.1%}
        • Generally trustworthy indicators
        
        **Assessment:**
        • Misinformation score: {misinfo_score:.1%}
        • Confidence: {confidence_score:.1%}
        • AI generation probability: {llm_prob:.1%}
        
        **Recommendation:** Content appears credible, but always verify important claims independently.
        """
    
    return explanation

def generate_youtube_explanation(
    content: str, 
    misinfo_score: float, 
    confidence_score: float, 
    llm_prob: float, 
    source: str
) -> str:
    """Generate explanation for YouTube content using the master prompt."""
    
    explanation = f"""
    **📹 YOUTUBE VIDEO ANALYSIS**
    
    **Video Analysis Results:**
    • Misinformation Score: {misinfo_score:.1%}
    • Confidence Level: {confidence_score:.1%}
    • LLM Origin Probability: {llm_prob:.1%}
    • Channel: {source}
    
    **Analysis Breakdown:**
    """
    
    if misinfo_score > 0.7:
        explanation += f"""
        🚨 **HIGH MISINFORMATION RISK DETECTED**
        
        This YouTube video shows strong indicators of potential misinformation:
        • Multiple red flags detected in transcript analysis
        • Visual content may contain manipulated elements
        • Channel history suggests unreliable content patterns
        
        **Key Concerns:**
        • Transcript misinformation: {misinfo_score:.1%}
        • Analysis confidence: {confidence_score:.1%}
        • Potential AI-generated script: {llm_prob:.1%}
        
        **Recommendation:** Exercise extreme caution. Verify claims through multiple reliable sources. Consider the channel's history of misinformation.
        """
    elif misinfo_score > 0.5:
        explanation += f"""
        ⚠️ **MODERATE MISINFORMATION RISK**
        
        This YouTube video shows some concerning indicators:
        • Moderate risk of misinformation in transcript
        • Some red flags present in content analysis
        • Mixed reliability indicators
        
        **Assessment Details:**
        • Transcript misinformation: {misinfo_score:.1%}
        • Confidence level: {confidence_score:.1%}
        • AI generation likelihood: {llm_prob:.1%}
        
        **Recommendation:** Approach with skepticism. Cross-reference information with established sources. Check channel credibility.
        """
    elif llm_prob > 0.7:
        explanation += f"""
        🤖 **LIKELY AI-GENERATED SCRIPT**
        
        This YouTube video appears to have an AI-generated script:
        • High probability of LLM origin: {llm_prob:.1%}
        • Script patterns suggest automated generation
        • May be part of coordinated content creation
        
        **Analysis:**
        • Misinformation score: {misinfo_score:.1%}
        • Confidence: {confidence_score:.1%}
        • AI generation probability: {llm_prob:.1%}
        
        **Recommendation:** Be aware this content may be artificially generated. Verify authenticity and check for coordinated posting patterns.
        """
    else:
        explanation += f"""
        ✅ **LIKELY CREDIBLE CONTENT**
        
        This YouTube video appears to be relatively reliable:
        • Low misinformation risk: {misinfo_score:.1%}
        • Low AI generation probability: {llm_prob:.1%}
        • Generally trustworthy indicators
        
        **Assessment:**
        • Transcript misinformation: {misinfo_score:.1%}
        • Confidence: {confidence_score:.1%}
        • AI generation probability: {llm_prob:.1%}
        
        **Recommendation:** Content appears credible, but always verify important claims independently.
        """
    
    return explanation

def generate_news_explanation(
    content: str, 
    misinfo_score: float, 
    confidence_score: float, 
    llm_prob: float, 
    source: str
) -> str:
    """Generate explanation for news content."""
    
    explanation = f"""
    **📰 NEWS ARTICLE ANALYSIS**
    
    **Article Analysis Results:**
    • Misinformation Score: {misinfo_score:.1%}
    • Confidence Level: {confidence_score:.1%}
    • LLM Origin Probability: {llm_prob:.1%}
    • Source: {source}
    
    **Analysis Breakdown:**
    """
    
    if misinfo_score > 0.7:
        explanation += f"""
        🚨 **HIGH MISINFORMATION RISK DETECTED**
        
        This article from **{source}** has been flagged with a **{misinfo_score:.1%} misinformation probability** 
        (confidence: {confidence_score:.1%}).
        
        **Key Concerns:**
        • The content shows strong indicators of potential misinformation
        • Multiple red flags detected in the analysis
        • Source credibility may be compromised
        
        **Analysis Details:**
        • Misinformation Score: {misinfo_score:.1%}
        • Confidence Level: {confidence_score:.1%}
        • Risk Assessment: HIGH
        
        **Recommendation:** Exercise extreme caution. Verify claims through multiple reliable sources before sharing or believing this content.
        """
    elif misinfo_score > 0.5:
        explanation += f"""
        ⚠️ **MODERATE MISINFORMATION RISK**
        
        This article from **{source}** shows some concerning indicators:
        • Moderate risk of misinformation: {misinfo_score:.1%}
        • Some red flags present in the analysis
        • Mixed reliability indicators
        
        **Assessment Details:**
        • Misinformation Score: {misinfo_score:.1%}
        • Confidence Level: {confidence_score:.1%}
        • Risk Assessment: MODERATE
        
        **Recommendation:** Approach with skepticism. Cross-reference information with established sources.
        """
    elif llm_prob > 0.7:
        explanation += f"""
        🤖 **LIKELY AI-GENERATED CONTENT**
        
        This article appears to be generated by an AI system:
        • High probability of LLM origin: {llm_prob:.1%}
        • Content patterns suggest automated generation
        • May be part of coordinated influence operations
        
        **Analysis:**
        • Misinformation score: {misinfo_score:.1%}
        • Confidence: {confidence_score:.1%}
        • AI generation probability: {llm_prob:.1%}
        
        **Recommendation:** Be aware this content may be artificially generated. Verify authenticity and sources.
        """
    else:
        explanation += f"""
        ✅ **LIKELY CREDIBLE CONTENT**
        
        This article appears to be relatively reliable:
        • Low misinformation risk: {misinfo_score:.1%}
        • Low AI generation probability: {llm_prob:.1%}
        • Generally trustworthy indicators
        
        **Assessment:**
        • Misinformation score: {misinfo_score:.1%}
        • Confidence: {confidence_score:.1%}
        • AI generation probability: {llm_prob:.1%}
        
        **Recommendation:** Content appears credible, but always verify important claims independently.
        """
    
    return explanation

def generate_openai_style_explanation(
    content: str,
    misinfo_score: float,
    confidence_score: float,
    llm_prob: float,
    content_type: str = "news",
    platform: str = "unknown"
) -> str:
    """Generate OpenAI-style explanation using the master prompt."""
    
    try:
        # Set up OpenAI client
        if not OPENAI_API_KEY:
            return generate_fallback_explanation(content, misinfo_score, confidence_score, llm_prob, content_type)
        
        openai.api_key = OPENAI_API_KEY
        
        # Create platform-specific master prompt
        if content_type == "reddit":
            master_prompt = f"""
            You are an expert in misinformation detection analyzing a Reddit post. Analyze this content based on model predictions and explain why it may be misinformation. Focus on logical fallacies, sensationalism, or signs of LLM generation. Use a calm, unbiased tone.

            Post Content:
            "{content[:1000]}"

            Model Outputs:
            Misinformation Score: {misinfo_score:.3f}
            Confidence Score: {confidence_score:.3f}
            LLM Origin Probability: {llm_prob:.3f}

            Your Explanation:
            """
        elif content_type == "video":
            master_prompt = f"""
            You are a misinformation auditor analyzing a YouTube video. Given the video transcript and model outputs, explain in simple, neutral language why this video may be misinformation. Focus on false claims, logical inconsistencies, or AI-like phrasing. Avoid political bias.

            Transcript:
            "{content[:1000]}"

            Model Outputs:
            Misinformation Score: {misinfo_score:.3f}
            Confidence Score: {confidence_score:.3f}
            LLM Origin Probability: {llm_prob:.3f}

            Provide your expert analysis below:
            """
        else:
            master_prompt = f"""
            You are an expert in misinformation detection analyzing a news article. Analyze this content based on model predictions and explain why it may be misinformation. Focus on logical fallacies, sensationalism, or signs of LLM generation. Use a calm, unbiased tone.

            Article Content:
            "{content[:1000]}"

            Model Outputs:
            Misinformation Score: {misinfo_score:.3f}
            Confidence Score: {confidence_score:.3f}
            LLM Origin Probability: {llm_prob:.3f}

            Your Explanation:
            """
        
        # Generate explanation using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert misinformation detection analyst. Provide clear, factual explanations without political bias."},
                {"role": "user", "content": master_prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating OpenAI explanation: {e}")
        return generate_fallback_explanation(content, misinfo_score, confidence_score, llm_prob, content_type)

def generate_fallback_explanation(
    content: str,
    misinfo_score: float,
    confidence_score: float,
    llm_prob: float,
    content_type: str = "news"
) -> str:
    """Generate fallback explanation when OpenAI is not available."""
    
    explanation = f"""
    **AI Analysis Summary:**
    
    Based on automated analysis, this {content_type} content shows the following characteristics:
    
    **Risk Assessment:**
    • Misinformation Probability: {misinfo_score:.1%}
    • Analysis Confidence: {confidence_score:.1%}
    • AI Generation Likelihood: {llm_prob:.1%}
    
    **Key Indicators:**
    """
    
    if misinfo_score > 0.7:
        explanation += "• High probability of containing false or misleading information\n"
        explanation += "• Multiple red flags detected in content analysis\n"
        explanation += "• Strong indicators of unreliable content\n"
    elif misinfo_score > 0.5:
        explanation += "• Moderate risk of misinformation\n"
        explanation += "• Some concerning patterns detected\n"
        explanation += "• Mixed reliability indicators\n"
    else:
        explanation += "• Low risk of misinformation\n"
        explanation += "• Generally reliable content patterns\n"
        explanation += "• Trustworthy indicators present\n"
    
    if llm_prob > 0.7:
        explanation += "• High likelihood of AI-generated content\n"
        explanation += "• Automated content patterns detected\n"
        explanation += "• May be part of coordinated operations\n"
    
    explanation += f"""
    **Recommendation:** {'Exercise extreme caution and verify claims independently.' if misinfo_score > 0.7 else 'Approach with healthy skepticism and cross-reference information.' if misinfo_score > 0.5 else 'Content appears relatively reliable, but always verify important claims.'}
    """
    
    return explanation

def analyze_feature_importance(content: str) -> Dict[str, Any]:
    """Analyze feature importance for the content."""
    try:
        features = {}
        
        # Text length
        features['text_length'] = len(content)
        
        # Sentiment indicators
        features['has_exclamation_marks'] = content.count('!') > 3
        features['has_caps_words'] = len([word for word in content.split() if word.isupper() and len(word) > 2]) > 2
        features['has_quotes'] = content.count('"') > 2
        
        # Content patterns
        features['has_numbers'] = any(c.isdigit() for c in content)
        features['has_urls'] = 'http' in content.lower() or 'www' in content.lower()
        features['has_hashtags'] = content.count('#') > 0
        
        # Sensationalist indicators
        sensationalist_words = ['breaking', 'shocking', 'exclusive', 'urgent', 'secret', 'conspiracy', 'cover-up']
        features['sensationalist_word_count'] = sum(1 for word in sensationalist_words if word.lower() in content.lower())
        
        # Credibility indicators
        features['has_citations'] = any(word in content.lower() for word in ['according to', 'study shows', 'research indicates', 'scientists say'])
        features['has_quotes'] = content.count('"') > 2
        
        return features
        
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return {}

def get_risk_level(misinfo_score: float, llm_prob: float) -> str:
    """Get risk level based on analysis scores."""
    try:
        # Calculate combined risk score
        risk_score = (misinfo_score * 0.7) + (llm_prob * 0.3)
        
        if risk_score > 0.8:
            return 'HIGH'
        elif risk_score > 0.5:
            return 'MODERATE'
        elif risk_score > 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
            
    except Exception as e:
        print(f"Error getting risk level: {e}")
        return 'UNKNOWN'

def generate_recommendations(
    misinfo_score: float, 
    llm_prob: float, 
    source: str,
    content_type: str = "news"
) -> List[str]:
    """Generate recommendations based on analysis results."""
    recommendations = []
    
    try:
        if misinfo_score > 0.7:
            recommendations.append("🚨 Exercise extreme caution - high misinformation risk detected")
            recommendations.append("🔍 Verify all claims through multiple reliable sources")
            recommendations.append("📰 Check fact-checking websites for this topic")
            recommendations.append("⚠️ Avoid sharing this content without verification")
        elif misinfo_score > 0.5:
            recommendations.append("⚠️ Approach with healthy skepticism")
            recommendations.append("🔍 Cross-reference information with established sources")
            recommendations.append("📊 Look for multiple perspectives on this topic")
            recommendations.append("🤔 Consider the source's track record")
        
        if llm_prob > 0.7:
            recommendations.append("🤖 Content may be AI-generated - verify authenticity")
            recommendations.append("🔗 Check for coordinated posting patterns")
            recommendations.append("👥 Look for human verification of claims")
        
        if content_type == "reddit":
            recommendations.append("📱 Check subreddit reputation and moderation")
            recommendations.append("👤 Review user's posting history")
        elif content_type == "video":
            recommendations.append("📺 Check channel's history and credibility")
            recommendations.append("🎬 Look for visual manipulation indicators")
        
        # Always include general recommendations
        recommendations.append("📚 Consult multiple reliable news sources")
        recommendations.append("🎯 Focus on facts over sensationalist language")
        recommendations.append("⏰ Be patient - wait for verified information")
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return ["Unable to generate specific recommendations"]

def format_explanation_for_display(explanation_data: Dict[str, Any]) -> str:
    """Format explanation data for display in Streamlit."""
    try:
        formatted_text = ""
        
        # Add explanation text
        if 'explanation_text' in explanation_data:
            formatted_text += explanation_data['explanation_text']
        
        # Add AI explanation if available
        if explanation_data.get('ai_explanation'):
            formatted_text += f"\n\n**🤖 AI Analysis:**\n{explanation_data['ai_explanation']}"
        
        # Add recommendations
        if explanation_data.get('recommendations'):
            formatted_text += f"\n\n**💡 Recommendations:**\n"
            for rec in explanation_data['recommendations']:
                formatted_text += f"• {rec}\n"
        
        # Add feature importance
        if explanation_data.get('feature_importance'):
            formatted_text += f"\n\n**🔍 Key Indicators:**\n"
            features = explanation_data['feature_importance']
            for feature, value in features.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    formatted_text += f"• {status} {feature.replace('_', ' ').title()}\n"
                else:
                    formatted_text += f"• {feature.replace('_', ' ').title()}: {value}\n"
        
        return formatted_text
        
    except Exception as e:
        print(f"Error formatting explanation: {e}")
        return "Error formatting explanation"

def generate_coordination_explanation(operation_data: Dict[str, Any]) -> str:
    """Generate explanation for coordinated influence operations."""
    try:
        explanation = f"""
        **🚨 COORDINATED INFLUENCE OPERATION DETECTED**
        
        **Operation Details:**
        • Topic: {operation_data.get('topic', 'Unknown')}
        • Affected Platforms: {', '.join(operation_data.get('affected_platforms', []))}
        • Affected Sources: {', '.join(operation_data.get('affected_sources', []))}
        • Total Flagged Content: {operation_data.get('total_flagged_content', 0)}
        • Average Misinformation Score: {operation_data.get('avg_misinfo_score', 0):.1%}
        • Severity Level: {operation_data.get('severity_level', 'Unknown')}
        
        **Analysis:**
        This coordinated operation shows patterns of misinformation across multiple platforms and sources, suggesting organized influence efforts rather than isolated incidents.
        
        **Recommendations:**
        • Monitor all affected platforms for similar content
        • Verify information through independent sources
        • Report coordinated misinformation to platform moderators
        • Stay informed through reliable fact-checking organizations
        """
        
        return explanation
        
    except Exception as e:
        print(f"Error generating coordination explanation: {e}")
        return "Error generating coordination explanation"

def generate_comprehensive_explanation_enhanced(
    content,
    misinfo_score,
    confidence_score,
    llm_prob,
    trust_score,
    credibility_score,
    source,
    content_type="news",
    use_openai=True
):
    """Enhanced comprehensive explanation generation."""
    try:
        # Generate base explanation
        explanation_data = generate_comprehensive_explanation(
            content, misinfo_score, confidence_score, llm_prob,
            trust_score, credibility_score, source, content_type
        )
        
        # Add enhanced features
        explanation_data['analysis_timestamp'] = datetime.now().isoformat()
        explanation_data['model_version'] = 'v2.0'
        explanation_data['confidence_metrics'] = {
            'misinfo_confidence': confidence_score,
            'llm_detection_confidence': 0.8,  # Placeholder
            'overall_confidence': (confidence_score + 0.8) / 2
        }
        
        # Add platform-specific analysis
        if content_type == "reddit":
            explanation_data['platform_analysis'] = {
                'subreddit_credibility': calculate_subreddit_credibility(source),
                'user_engagement_metrics': analyze_reddit_engagement(content),
                'moderation_indicators': check_moderation_activity(source)
            }
        elif content_type == "video":
            explanation_data['platform_analysis'] = {
                'channel_credibility': calculate_channel_credibility(source),
                'video_quality_metrics': analyze_video_quality(content),
                'viewer_engagement': analyze_viewer_engagement(content)
            }
        
        return explanation_data
        
    except Exception as e:
        print(f"Error in enhanced explanation generation: {e}")
        return {
            'error': str(e),
            'explanation_type': 'error',
            'explanation_text': f"Error generating enhanced explanation: {str(e)}"
        }

def calculate_subreddit_credibility(subreddit_name: str) -> float:
    """Calculate credibility score for a subreddit."""
    try:
        # This would typically query a database of subreddit reputation scores
        # For now, return a placeholder score
        return 0.6
    except Exception as e:
        print(f"Error calculating subreddit credibility: {e}")
        return 0.5

def calculate_channel_credibility(channel_name: str) -> float:
    """Calculate credibility score for a YouTube channel."""
    try:
        # This would typically query a database of channel reputation scores
        # For now, return a placeholder score
        return 0.6
    except Exception as e:
        print(f"Error calculating channel credibility: {e}")
        return 0.5

def analyze_reddit_engagement(content: str) -> Dict[str, Any]:
    """Analyze Reddit engagement metrics."""
    try:
        return {
            'upvote_ratio': 0.8,  # Placeholder
            'comment_count': len(content.split()) // 10,  # Rough estimate
            'engagement_score': 0.7  # Placeholder
        }
    except Exception as e:
        print(f"Error analyzing Reddit engagement: {e}")
        return {}

def analyze_video_quality(content: str) -> Dict[str, Any]:
    """Analyze video quality metrics."""
    try:
        return {
            'transcript_quality': 0.8,  # Placeholder
            'audio_quality': 0.7,  # Placeholder
            'visual_quality': 0.6  # Placeholder
        }
    except Exception as e:
        print(f"Error analyzing video quality: {e}")
        return {}

def analyze_viewer_engagement(content: str) -> Dict[str, Any]:
    """Analyze viewer engagement metrics."""
    try:
        return {
            'view_count': 1000,  # Placeholder
            'like_ratio': 0.8,  # Placeholder
            'comment_engagement': 0.6  # Placeholder
        }
    except Exception as e:
        print(f"Error analyzing viewer engagement: {e}")
        return {}

def check_moderation_activity(subreddit_name: str) -> Dict[str, Any]:
    """Check moderation activity for a subreddit."""
    try:
        return {
            'moderator_count': 5,  # Placeholder
            'moderation_activity': 'active',  # Placeholder
            'rule_enforcement': 0.7  # Placeholder
        }
    except Exception as e:
        print(f"Error checking moderation activity: {e}")
        return {}

def format_explanation_for_display_enhanced(explanation_data: Dict[str, Any], show_ai_analysis: bool = True) -> str:
    """Enhanced formatting for display with additional features."""
    try:
        formatted_text = format_explanation_for_display(explanation_data)
        
        # Add platform-specific analysis
        if 'platform_analysis' in explanation_data:
            formatted_text += f"\n\n**📊 Platform Analysis:**\n"
            platform_data = explanation_data['platform_analysis']
            
            for key, value in platform_data.items():
                if isinstance(value, dict):
                    formatted_text += f"• {key.replace('_', ' ').title()}:\n"
                    for sub_key, sub_value in value.items():
                        formatted_text += f"  - {sub_key.replace('_', ' ').title()}: {sub_value}\n"
                else:
                    formatted_text += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        # Add confidence metrics
        if 'confidence_metrics' in explanation_data:
            formatted_text += f"\n\n**🎯 Confidence Metrics:**\n"
            metrics = explanation_data['confidence_metrics']
            for key, value in metrics.items():
                formatted_text += f"• {key.replace('_', ' ').title()}: {value:.1%}\n"
        
        return formatted_text
        
    except Exception as e:
        print(f"Error in enhanced formatting: {e}")
        return format_explanation_for_display(explanation_data) 