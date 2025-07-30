import numpy as np
from datetime import datetime, timedelta
try:
    from config import TRUST_WEIGHTS, TRUST_THRESHOLD
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TRUST_WEIGHTS, TRUST_THRESHOLD

class TrustCredibilityAnalyzer:
    def __init__(self):
        """Initialize trust and credibility analyzer."""
        self.trust_weights = TRUST_WEIGHTS
        self.trust_threshold = TRUST_THRESHOLD
        
        # Known reliable sources
        self.reliable_sources = {
            'reuters.com': 0.9,
            'ap.org': 0.9,
            'bbc.com': 0.85,
            'npr.org': 0.85,
            'nytimes.com': 0.8,
            'washingtonpost.com': 0.8,
            'wsj.com': 0.8,
            'cnn.com': 0.75,
            'abcnews.go.com': 0.75,
            'cbsnews.com': 0.75,
            'nbcnews.com': 0.75,
            'foxnews.com': 0.7,
            'usatoday.com': 0.7,
            'latimes.com': 0.75,
            'chicagotribune.com': 0.7,
            'bostonglobe.com': 0.75
        }
        
        # Known unreliable sources
        self.unreliable_sources = {
            'infowars.com': 0.1,
            'breitbart.com': 0.3,
            'dailycaller.com': 0.3,
            'theblaze.com': 0.3,
            'naturalnews.com': 0.1,
            'prisonplanet.com': 0.1,
            'beforeitsnews.com': 0.1,
            'yournewswire.com': 0.1
        }
    
    def calculate_source_reliability(self, source_url):
        """Calculate source reliability score based on known sources."""
        if not source_url:
            return 0.5
        
        # Extract domain from URL
        domain = self._extract_domain(source_url)
        
        # Check against known reliable sources
        if domain in self.reliable_sources:
            return self.reliable_sources[domain]
        
        # Check against known unreliable sources
        if domain in self.unreliable_sources:
            return self.unreliable_sources[domain]
        
        # Default score for unknown sources
        return 0.5
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        try:
            if url.startswith('http'):
                from urllib.parse import urlparse
                parsed = urlparse(url)
                return parsed.netloc.lower()
            else:
                return url.lower()
        except:
            return url.lower()
    
    def calculate_author_credibility(self, author_info):
        """Calculate author credibility score."""
        if not author_info:
            return 0.5
        
        credibility_score = 0.5  # Base score
        
        # Check for verified status indicators
        verified_indicators = ['verified', 'official', 'expert', 'professor', 'dr.', 'phd']
        author_lower = author_info.lower()
        
        for indicator in verified_indicators:
            if indicator in author_lower:
                credibility_score += 0.1
        
        # Check for suspicious indicators
        suspicious_indicators = ['anonymous', 'unknown', 'unverified', 'fake']
        for indicator in suspicious_indicators:
            if indicator in author_lower:
                credibility_score -= 0.2
        
        return max(0.0, min(1.0, credibility_score))
    
    def calculate_cross_check_consistency(self, content_metadata, similar_content):
        """Calculate consistency with other similar content."""
        if not similar_content or not isinstance(similar_content, (list, tuple)):
            return 0.5
        
        consistency_scores = []
        
        try:
            for content in similar_content:
                if not isinstance(content, dict):
                    continue
                    
                # Compare key claims or facts
                if 'title' in content_metadata and 'title' in content:
                    title1 = content_metadata['title']
                    title2 = content['title']
                    if title1 and title2:  # Ensure both titles are not None
                        title_similarity = self._calculate_text_similarity(title1, title2)
                        consistency_scores.append(title_similarity)
                
                # Compare sources
                if 'source' in content_metadata and 'source' in content:
                    source1 = content_metadata['source']
                    source2 = content['source']
                    if source1 and source2:  # Ensure both sources are not None
                        source_similarity = 1.0 if source1 == source2 else 0.0
                        consistency_scores.append(source_similarity)
        except Exception as e:
            print(f"Error in cross-check consistency: {e}")
            return 0.5
        
        if consistency_scores and len(consistency_scores) > 0:
            try:
                return float(np.mean(consistency_scores))
            except Exception as e:
                print(f"Error calculating mean: {e}")
                return 0.5
        else:
            return 0.5
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Ensure both texts are strings
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_content_quality(self, content_text, content_length):
        """Calculate content quality score."""
        if not content_text:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length factor
        if content_length > 500:
            quality_score += 0.2
        elif content_length > 200:
            quality_score += 0.1
        elif content_length < 50:
            quality_score -= 0.2
        
        # Check for citations and references
        citation_indicators = ['according to', 'study shows', 'research indicates', 'sources:', 'references:']
        content_lower = content_text.lower()
        
        try:
            citation_count = sum(1 for indicator in citation_indicators if indicator in content_lower)
            quality_score += min(0.2, citation_count * 0.05)
        except Exception as e:
            print(f"Error calculating citation count: {e}")
        
        # Check for balanced reporting
        balanced_indicators = ['however', 'on the other hand', 'alternatively', 'meanwhile']
        try:
            balanced_count = sum(1 for indicator in balanced_indicators if indicator in content_lower)
            quality_score += min(0.1, balanced_count * 0.02)
        except Exception as e:
            print(f"Error calculating balanced count: {e}")
        
        return max(0.0, min(1.0, quality_score))
    
    def calculate_temporal_freshness(self, publish_date):
        """Calculate temporal freshness score."""
        if not publish_date:
            return 0.5
        
        try:
            if isinstance(publish_date, str):
                # Parse date string
                publish_dt = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
            else:
                publish_dt = publish_date
            
            now = datetime.now()
            age_days = (now - publish_dt).days
            
            # Calculate freshness score (newer = higher score)
            if age_days <= 1:
                freshness_score = 1.0
            elif age_days <= 7:
                freshness_score = 0.9
            elif age_days <= 30:
                freshness_score = 0.8
            elif age_days <= 90:
                freshness_score = 0.7
            elif age_days <= 365:
                freshness_score = 0.6
            else:
                freshness_score = 0.5
            
            return freshness_score
            
        except Exception as e:
            print(f"Error calculating temporal freshness: {e}")
            return 0.5
    
    def calculate_trust_score(self, content_metadata, similar_content=None):
        """Calculate overall trust score for content."""
        if not content_metadata or not isinstance(content_metadata, dict):
            return {
                'trust_score': 0.5,
                'trust_components': {},
                'is_trustworthy': False
            }
        
        trust_components = {}
        
        # Source reliability
        source_url = content_metadata.get('url', '')
        trust_components['source_reliability'] = self.calculate_source_reliability(source_url)
        
        # Author credibility
        author = content_metadata.get('author', '')
        trust_components['author_credibility'] = self.calculate_author_credibility(author)
        
        # Cross-check consistency
        trust_components['cross_check_consistency'] = self.calculate_cross_check_consistency(
            content_metadata, similar_content or []
        )
        
        # Content quality
        content_text = content_metadata.get('content', '')
        content_length = len(content_text) if content_text else 0
        trust_components['content_quality'] = self.calculate_content_quality(content_text, content_length)
        
        # Temporal freshness
        publish_date = content_metadata.get('published_date')
        trust_components['temporal_freshness'] = self.calculate_temporal_freshness(publish_date)
        
        # Calculate weighted trust score
        weighted_score = sum(
            trust_components[component] * self.trust_weights[component]
            for component in self.trust_weights.keys()
            if component in trust_components
        )
        
        return {
            'trust_score': weighted_score,
            'trust_components': trust_components,
            'is_trustworthy': weighted_score >= self.trust_threshold
        }
    
    def calculate_credibility_score(self, content_metadata, analysis_results):
        """Calculate credibility score based on analysis results."""
        if not analysis_results or not isinstance(analysis_results, dict):
            return 0.5
        
        credibility_score = 0.5  # Base score
        
        # Adjust based on misinformation score
        misinfo_score = analysis_results.get('misinformation_score', 0.5)
        credibility_score -= misinfo_score * 0.3  # Reduce credibility if high misinformation
        
        # Adjust based on confidence score
        confidence_score = analysis_results.get('confidence_score', 0.5)
        credibility_score += confidence_score * 0.2  # Increase credibility if high confidence
        
        # Adjust based on LLM origin probability
        llm_prob = analysis_results.get('llm_origin_probability', 0.5)
        credibility_score -= llm_prob * 0.1  # Slightly reduce credibility if likely LLM-generated
        
        # Adjust based on entropy (higher entropy = more natural)
        entropy_score = analysis_results.get('entropy_score', 0.0)
        credibility_score += min(0.1, entropy_score * 0.01)  # Small boost for natural text
        
        return max(0.0, min(1.0, credibility_score))
    
    def generate_trust_explanation(self, trust_result):
        """Generate human-readable explanation of trust score."""
        components = trust_result['trust_components']
        explanations = []
        
        if components.get('source_reliability', 0) > 0.7:
            explanations.append("✅ Source is known to be reliable")
        elif components.get('source_reliability', 0) < 0.3:
            explanations.append("⚠️ Source has questionable reliability")
        
        if components.get('author_credibility', 0) > 0.7:
            explanations.append("✅ Author appears credible")
        elif components.get('author_credibility', 0) < 0.3:
            explanations.append("⚠️ Author credibility is questionable")
        
        if components.get('content_quality', 0) > 0.7:
            explanations.append("✅ Content appears well-researched")
        elif components.get('content_quality', 0) < 0.3:
            explanations.append("⚠️ Content quality is low")
        
        if components.get('temporal_freshness', 0) > 0.8:
            explanations.append("✅ Content is recent")
        elif components.get('temporal_freshness', 0) < 0.5:
            explanations.append("⚠️ Content may be outdated")
        
        if not explanations:
            explanations.append("📊 Trust assessment is neutral")
        
        return explanations

# Global analyzer instance
trust_analyzer = TrustCredibilityAnalyzer()

def calculate_trust_score(content_metadata, similar_content=None):
    """Convenience function to calculate trust score."""
    return trust_analyzer.calculate_trust_score(content_metadata, similar_content)

def calculate_credibility_score(content_metadata, analysis_results):
    """Convenience function to calculate credibility score."""
    return trust_analyzer.calculate_credibility_score(content_metadata, analysis_results)

def generate_trust_explanation(trust_result):
    """Convenience function to generate trust explanation."""
    return trust_analyzer.generate_trust_explanation(trust_result) 