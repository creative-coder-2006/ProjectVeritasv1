import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Any, Optional
import json
import openai
try:
    from config import OPENAI_API_KEY
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import OPENAI_API_KEY
import re
# Custom sentiment analysis implementation
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import random
from datetime import datetime

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Global variables for models
misinfo_pipeline = None
llm_detection_pipeline = None
sentiment_analyzer = None
tokenizer = None
model = None
# In veritasai/analyzer/text_analysis.py

# ... (other imports)
try:
    from config import OPENAI_API_KEY, MISINFORMATION_MODEL, FALLBACK_MISINFORMATION_MODEL
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import OPENAI_API_KEY, MISINFORMATION_MODEL, FALLBACK_MISINFORMATION_MODEL

# ...

def load_misinformation_model():
    """Load the misinformation detection model."""
    global misinfo_pipeline
    
    if misinfo_pipeline is None:
        try:
            print(f"Loading misinformation detection model: {MISINFORMATION_MODEL}")
            misinfo_pipeline = pipeline(
                "text-classification",
                model=MISINFORMATION_MODEL,
                device=0 if torch.cuda.is_available() else -1
            )
            print("Misinformation model loaded successfully")
        except Exception as e:
            print(f"Error loading primary misinformation model: {e}")
            print(f"Attempting to load fallback model: {FALLBACK_MISINFORMATION_MODEL}")
            try:
                misinfo_pipeline = pipeline(
                    "text-classification",
                    model=FALLBACK_MISINFORMATION_MODEL,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Fallback misinformation model loaded successfully")
            except Exception as e_fallback:
                print(f"Error loading fallback misinformation model: {e_fallback}")
                import traceback
                traceback.print_exc()
                return False
    
    return True

def load_llm_detection_model():
    """Load the LLM origin detection model (DetectGPT-like)."""
    global llm_detection_pipeline
    
    if llm_detection_pipeline is None:
        try:
            print("Loading LLM detection model...")
            # Using a model that can detect AI-generated text
            llm_detection_pipeline = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Alternative model for LLM detection
                device=0 if torch.cuda.is_available() else -1
            )
            print("LLM detection model loaded successfully")
        except Exception as e:
            print(f"Error loading LLM detection model: {e}")
            return False
    
    return True

def load_sentiment_analyzer():
    """Load the sentiment analyzer (custom implementation)."""
    global sentiment_analyzer
    
    if sentiment_analyzer is None:
        try:
            # Custom sentiment analyzer using NLTK
            sentiment_analyzer = CustomSentimentAnalyzer()
        except Exception as e:
            print(f"Error loading sentiment analyzer: {e}")
            return False
    
    return True

class CustomSentimentAnalyzer:
    """Custom sentiment analyzer using NLTK and simple heuristics."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Positive and negative word lists
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'brilliant', 'outstanding', 'superb', 'terrific', 'perfect',
            'love', 'like', 'enjoy', 'happy', 'joy', 'pleasure', 'delight',
            'success', 'win', 'victory', 'achievement', 'progress', 'improvement',
            'beautiful', 'gorgeous', 'stunning', 'magnificent', 'splendid',
            'helpful', 'useful', 'beneficial', 'valuable', 'important', 'essential'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'disgusting',
            'hate', 'dislike', 'loathe', 'despise', 'abhor', 'detest',
            'sad', 'unhappy', 'miserable', 'depressed', 'gloomy', 'melancholy',
            'failure', 'lose', 'defeat', 'loss', 'disaster', 'catastrophe',
            'ugly', 'hideous', 'repulsive', 'revolting', 'nauseating',
            'harmful', 'dangerous', 'risky', 'threatening', 'damaging', 'destructive'
        }
        
        # Negation words
        self.negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 
                              'neither', 'nowhere', 'hardly', 'barely', 'scarcely'}
        
        # Intensifier words
        self.intensifier_words = {'very', 'really', 'extremely', 'incredibly', 
                                 'absolutely', 'completely', 'totally', 'utterly'}
    
    def polarity_scores(self, text):
        """Calculate sentiment polarity scores similar to VADER."""
        try:
            # Tokenize text
            words = word_tokenize(text.lower())
            
            # Initialize scores
            positive_score = 0
            negative_score = 0
            neutral_score = 0
            
            # Track negation and intensifiers
            negated = False
            intensifier_count = 0
            
            for i, word in enumerate(words):
                # Check for negation
                if word in self.negation_words:
                    negated = not negated
                    continue
                
                # Check for intensifiers
                if word in self.intensifier_words:
                    intensifier_count += 1
                    continue
                
                # Check word sentiment
                if word in self.positive_words:
                    score = 1.0
                    if negated:
                        score = -0.5
                        negative_score += score
                    else:
                        positive_score += score
                    negated = False
                    
                elif word in self.negative_words:
                    score = -1.0
                    if negated:
                        score = 0.5
                        positive_score += score
                    else:
                        negative_score += score
                    negated = False
                    
                else:
                    neutral_score += 1
                    negated = False
                
                # Apply intensifier effect
                if intensifier_count > 0:
                    if score > 0:
                        positive_score += (intensifier_count * 0.3)
                    elif score < 0:
                        negative_score += (intensifier_count * 0.3)
                    intensifier_count = 0
            
            # Calculate compound score
            total_words = len(words)
            if total_words == 0:
                return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
            
            # Normalize scores
            pos_ratio = positive_score / total_words
            neg_ratio = abs(negative_score) / total_words
            neu_ratio = neutral_score / total_words
            
            # Calculate compound score (VADER-like formula)
            compound = pos_ratio - neg_ratio
            compound = max(-1.0, min(1.0, compound))
            
            return {
                'compound': compound,
                'pos': pos_ratio,
                'neg': neg_ratio,
                'neu': neu_ratio
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

def calculate_confidence_score(predictions: List[Dict]) -> float:
    """Calculate confidence score from model predictions."""
    try:
        if not predictions:
            return 0.0
        
        # Extract scores and convert to probabilities
        scores = [pred.get('score', 0.0) for pred in predictions]
        
        # Use softmax to get proper probabilities
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Confidence is the maximum probability
        confidence = float(np.max(probabilities))
        
        return confidence
    except Exception as e:
        print(f"Error calculating confidence score: {e}")
        return 0.5  # Default confidence

def detect_llm_origin(text: str) -> float:
    """Detect if text was likely generated by an LLM using enhanced methods."""
    try:
        if not load_llm_detection_model():
            return 0.5  # Default probability
        
        # Method 1: Use the model to detect AI-generated text patterns
        result = llm_detection_pipeline(text[:512])  # Limit text length
        
        # Method 2: Statistical analysis for LLM patterns
        statistical_score = analyze_llm_statistical_patterns(text)
        
        # Method 3: Perplexity-based detection
        perplexity_score = calculate_text_perplexity(text)
        
        # Combine scores
        model_score = result[0].get('score', 0.5) if isinstance(result, list) and len(result) > 0 else 0.5
        
        # Weighted combination
        final_score = (0.4 * model_score + 0.3 * statistical_score + 0.3 * perplexity_score)
        
        return min(1.0, max(0.0, final_score))
            
    except Exception as e:
        print(f"Error in LLM origin detection: {e}")
        return 0.5

def analyze_llm_statistical_patterns(text: str) -> float:
    """Analyze statistical patterns that indicate LLM generation."""
    try:
        if len(text) < 50:
            return 0.5
        
        # Tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove stopwords for analysis
        stop_words = set(stopwords.words('english'))
        content_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Calculate various metrics
        metrics = {}
        
        # 1. Sentence length consistency (LLMs tend to be very consistent)
        sentence_lengths = [len(sent.split()) for sent in sentences]
        metrics['sentence_length_std'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # 2. Word repetition patterns
        word_freq = {}
        for word in content_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate repetition score
        total_words = len(content_words)
        unique_words = len(word_freq)
        repetition_ratio = unique_words / total_words if total_words > 0 else 0
        metrics['repetition_ratio'] = repetition_ratio
        
        # 3. Vocabulary diversity
        metrics['vocabulary_diversity'] = len(set(content_words)) / len(content_words) if content_words else 0
        
        # 4. Punctuation patterns
        punctuation_count = len(re.findall(r'[.!?]', text))
        metrics['punctuation_density'] = punctuation_count / len(sentences) if sentences else 0
        
        # 5. Capitalization patterns
        caps_count = len(re.findall(r'[A-Z]', text))
        total_chars = len(re.findall(r'[a-zA-Z]', text))
        metrics['capitalization_ratio'] = caps_count / total_chars if total_chars > 0 else 0
        
        # Calculate LLM probability based on metrics
        llm_indicators = 0
        
        # Low sentence length variation indicates LLM
        if metrics['sentence_length_std'] < 3.0:
            llm_indicators += 1
        
        # High vocabulary diversity indicates LLM
        if metrics['vocabulary_diversity'] > 0.8:
            llm_indicators += 1
        
        # Consistent punctuation indicates LLM
        if 0.5 < metrics['punctuation_density'] < 2.0:
            llm_indicators += 1
        
        # Normal capitalization indicates LLM
        if 0.05 < metrics['capitalization_ratio'] < 0.15:
            llm_indicators += 1
        
        # Convert to probability
        llm_probability = llm_indicators / 4.0
        
        return llm_probability
        
    except Exception as e:
        print(f"Error analyzing LLM statistical patterns: {e}")
        return 0.5

def calculate_text_perplexity(text: str) -> float:
    """Calculate text perplexity as a measure of LLM generation."""
    try:
        if len(text) < 20:
            return 0.5
        
        # Simple n-gram perplexity calculation
        words = word_tokenize(text.lower())
        
        if len(words) < 3:
            return 0.5
        
        # Calculate bigram probabilities
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_freq = {}
        
        for bigram in bigrams:
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
        
        # Calculate perplexity
        total_bigrams = len(bigrams)
        log_prob = 0
        
        for bigram in bigrams:
            prob = bigram_freq[bigram] / total_bigrams
            log_prob += np.log(prob + 1e-10)  # Add small epsilon to avoid log(0)
        
        perplexity = np.exp(-log_prob / total_bigrams)
        
        # Normalize perplexity to 0-1 range (lower perplexity = more likely LLM)
        # Typical human text has higher perplexity than LLM text
        normalized_perplexity = max(0.0, min(1.0, 1.0 - (perplexity / 1000.0)))
        
        return normalized_perplexity
        
    except Exception as e:
        print(f"Error calculating text perplexity: {e}")
        return 0.5

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze a single text for misinformation and LLM origin."""
    if not text or len(text.strip()) < 10:
        return {
            'misinformation_score': 0.0,
            'confidence_score': 0.0,
            'llm_origin_probability': 0.0,
            'error': 'Text too short for analysis'
        }
    
    try:
        # Load models
        if not load_misinformation_model():
            return {
                'misinformation_score': 0.0,
                'confidence_score': 0.0,
                'llm_origin_probability': 0.0,
                'error': 'Failed to load misinformation model'
            }
        
        # Analyze for misinformation
        misinfo_result = misinfo_pipeline(text[:512])  # Limit text length
        
        # Calculate scores
        misinfo_score = misinfo_result[0].get('score', 0.0) if misinfo_result else 0.0
        confidence_score = calculate_confidence_score(misinfo_result)
        
        # Detect LLM origin
        llm_prob = detect_llm_origin(text)
        
        # Additional analysis
        sentiment_score = analyze_sentiment(text)
        readability_score = calculate_readability(text)
        
        return {
            'misinformation_score': misinfo_score,
            'confidence_score': confidence_score,
            'llm_origin_probability': llm_prob,
            'sentiment_score': sentiment_score,
            'readability_score': readability_score,
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text))
        }
        
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return {
            'misinformation_score': 0.0,
            'confidence_score': 0.0,
            'llm_origin_probability': 0.0,
            'error': str(e)
        }

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment of the text."""
    try:
        if not load_sentiment_analyzer():
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
        
        return sentiment_analyzer.polarity_scores(text)
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}

def calculate_readability(text: str) -> float:
    """Calculate readability score using Flesch Reading Ease."""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        syllables = count_syllables(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        
        # Normalize to 0-1 range
        normalized_score = max(0.0, min(1.0, flesch_score / 100.0))
        
        return normalized_score
        
    except Exception as e:
        print(f"Error calculating readability: {e}")
        return 0.5

def count_syllables(text: str) -> int:
    """Count syllables in text (simplified method)."""
    try:
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    except Exception as e:
        print(f"Error counting syllables: {e}")
        return len(text.split())  # Fallback

def analyze_multiple_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple texts for misinformation and LLM origin."""
    results = []
    
    for text in texts:
        result = analyze_text(text)
        results.append(result)
    
    return results

def calculate_trust_score(misinfo_score: float, confidence_score: float) -> float:
    """Calculate trust score based on misinformation and confidence scores."""
    try:
        # Trust score = 100 - (misinfo_score * (1 - confidence_score) * 100)
        trust_score = 100 - (misinfo_score * (1 - confidence_score) * 100)
        return max(0.0, min(100.0, trust_score))
    except Exception as e:
        print(f"Error calculating trust score: {e}")
        return 50.0

def calculate_credibility_score(source: str, misinfo_score: float, llm_prob: float) -> float:
    """Calculate credibility score based on source and analysis results."""
    try:
        # Base credibility score
        base_score = 50.0
        
        # Adjust based on misinformation score
        misinfo_penalty = misinfo_score * 30.0
        base_score -= misinfo_penalty
        
        # Adjust based on LLM probability
        llm_penalty = llm_prob * 20.0
        base_score -= llm_penalty
        
        # Source-specific adjustments
        source_lower = source.lower()
        
        # Known reliable sources
        reliable_sources = ['reuters', 'ap', 'bbc', 'cnn', 'npr', 'pbs', 'nyt', 'washington post']
        if any(reliable in source_lower for reliable in reliable_sources):
            base_score += 10.0
        
        # Known unreliable sources
        unreliable_sources = ['infowars', 'breitbart', 'daily stormer', 'stormfront']
        if any(unreliable in source_lower for unreliable in unreliable_sources):
            base_score -= 20.0
        
        return max(0.0, min(100.0, base_score))
        
    except Exception as e:
        print(f"Error calculating credibility score: {e}")
        return 50.0

def generate_synthetic_misinformation(topic: str, num_samples: int = 3) -> List[Dict[str, str]]:
    """Generate synthetic misinformation samples for contrastive evaluation."""
    try:
        synthetic_samples = []
        
        # Template-based generation
        templates = [
            f"BREAKING: Shocking new evidence reveals that {topic} is actually a government conspiracy designed to control the population. Sources say...",
            f"Scientists are hiding the truth about {topic}! New research shows completely different results than what the mainstream media reports.",
            f"Exclusive: Anonymous whistleblower exposes the real story behind {topic}. What they don't want you to know...",
            f"URGENT: {topic} has been linked to dangerous side effects that the FDA is covering up. Read this before it's deleted!",
            f"Conspiracy theorists were right about {topic}! New documents prove everything they've been saying for years."
        ]
        
        for i in range(min(num_samples, len(templates))):
            sample = {
                'title': f"Synthetic Misinformation Sample {i+1}",
                'content': templates[i],
                'source': 'synthetic_generation',
                'generated_by': 'template_based',
                'topic': topic,
                'is_synthetic': True
            }
            synthetic_samples.append(sample)
        
        return synthetic_samples
        
    except Exception as e:
        print(f"Error generating synthetic misinformation: {e}")
        return []

def comprehensive_text_analysis(text: str, source: str = "Unknown") -> Dict[str, Any]:
    """Perform comprehensive text analysis including all metrics."""
    try:
        # Basic analysis
        basic_analysis = analyze_text(text)
        
        if 'error' in basic_analysis:
            return basic_analysis
        
        # Calculate derived scores
        trust_score = calculate_trust_score(
            basic_analysis['misinformation_score'],
            basic_analysis['confidence_score']
        )
        
        credibility_score = calculate_credibility_score(
            source,
            basic_analysis['misinformation_score'],
            basic_analysis['llm_origin_probability']
        )
        
        # Risk assessment
        risk_level = assess_risk_level(
            basic_analysis['misinformation_score'],
            basic_analysis['llm_origin_probability']
        )
        
        # Feature importance analysis
        feature_importance = analyze_feature_importance(text)
        
        return {
            **basic_analysis,
            'trust_score': trust_score,
            'credibility_score': credibility_score,
            'risk_level': risk_level,
            'source': source,
            'feature_importance': feature_importance,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in comprehensive text analysis: {e}")
        return {
            'error': str(e),
            'misinformation_score': 0.0,
            'confidence_score': 0.0,
            'llm_origin_probability': 0.0,
            'trust_score': 50.0,
            'credibility_score': 50.0,
            'risk_level': 'unknown'
        }

def assess_risk_level(misinfo_score: float, llm_prob: float) -> str:
    """Assess risk level based on analysis scores."""
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
        print(f"Error assessing risk level: {e}")
        return 'UNKNOWN'

def analyze_feature_importance(text: str) -> Dict[str, float]:
    """Analyze feature importance for the text."""
    try:
        features = {}
        
        # Text length importance
        features['text_length'] = min(1.0, len(text) / 1000.0)
        
        # Sentiment importance
        sentiment = analyze_sentiment(text)
        features['sentiment_polarity'] = abs(sentiment.get('compound', 0.0))
        
        # Readability importance
        features['readability'] = calculate_readability(text)
        
        # Vocabulary diversity
        words = word_tokenize(text.lower())
        unique_words = len(set(words))
        features['vocabulary_diversity'] = unique_words / len(words) if words else 0.0
        
        # Punctuation patterns
        punctuation_count = len(re.findall(r'[.!?]', text))
        features['punctuation_density'] = punctuation_count / len(sent_tokenize(text)) if text else 0.0
        
        return features
        
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return {}

def analyze_with_synthetic_comparison(text: str, source: str, topic: str) -> Dict[str, Any]:
    """Analyze text with synthetic misinformation comparison."""
    try:
        # Analyze the original text
        original_analysis = comprehensive_text_analysis(text, source)
        
        # Generate synthetic samples
        synthetic_samples = generate_synthetic_misinformation(topic, 3)
        
        # Analyze synthetic samples
        synthetic_analyses = []
        for sample in synthetic_samples:
            analysis = comprehensive_text_analysis(sample['content'], sample['source'])
            synthetic_analyses.append(analysis)
        
        # Calculate contrastive metrics
        avg_synthetic_misinfo = np.mean([a.get('misinformation_score', 0.0) for a in synthetic_analyses])
        contrastive_score = original_analysis.get('misinformation_score', 0.0) / avg_synthetic_misinfo if avg_synthetic_misinfo > 0 else 0.0
        
        return {
            'original_analysis': original_analysis,
            'synthetic_samples': synthetic_samples,
            'synthetic_analyses': synthetic_analyses,
            'contrastive_score': contrastive_score,
            'topic': topic
        }
        
    except Exception as e:
        print(f"Error in synthetic comparison analysis: {e}")
        return {
            'error': str(e),
            'original_analysis': comprehensive_text_analysis(text, source)
        } 