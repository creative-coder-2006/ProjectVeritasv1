from faster_whisper import WhisperModel
import torch
import numpy as np
from datetime import datetime
try:
    from config import WHISPER_MODEL
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import WHISPER_MODEL
from .text_analysis import analyze_text

class AudioAnalyzer:
    def __init__(self):
        """Initialize audio analysis models."""
        self.whisper_model = None
        self._load_models()
    
    def _load_models(self):
        """Load the required models."""
        try:
            print(f"Loading Whisper model: {WHISPER_MODEL}")
            self.whisper_model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper_model = None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file to text."""
        if not self.whisper_model:
            return {
                'transcript': '',
                'confidence': 0.0,
                'error': 'Whisper model not loaded'
            }
        
        try:
            # Transcribe audio
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
            
            # Combine all segments into one transcript
            transcript = " ".join([segment.text for segment in segments])
            
            # Calculate average confidence
            confidences = [segment.avg_logprob for segment in segments if segment.avg_logprob is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Convert segments to list format
            segments_list = [
                {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'avg_logprob': segment.avg_logprob
                }
                for segment in segments
            ]
            
            return {
                'transcript': transcript,
                'confidence': avg_confidence,
                'segments': segments_list,
                'language': info.language
            }
            
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return {
                'transcript': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_audio_quality(self, audio_path):
        """Analyze audio quality metrics."""
        try:
            import librosa
            
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate audio quality metrics
            duration = len(y) / sr
            rms_energy = np.sqrt(np.mean(y**2))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Calculate signal-to-noise ratio (approximation)
            noise_floor = np.percentile(np.abs(y), 10)
            signal_level = np.percentile(np.abs(y), 90)
            snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            return {
                'duration': duration,
                'rms_energy': rms_energy,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate,
                'snr': snr,
                'sample_rate': sr
            }
            
        except Exception as e:
            print(f"Error analyzing audio quality: {e}")
            return {
                'duration': 0.0,
                'rms_energy': 0.0,
                'spectral_centroid': 0.0,
                'zero_crossing_rate': 0.0,
                'snr': 0.0,
                'sample_rate': 0
            }
    
    def detect_audio_anomalies(self, audio_path):
        """Detect potential audio anomalies that might indicate manipulation."""
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            anomalies = []
            
            # Check for sudden volume changes (potential cuts)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Detect sudden changes in RMS
            rms_diff = np.diff(rms)
            threshold = np.std(rms_diff) * 2
            
            sudden_changes = np.where(np.abs(rms_diff) > threshold)[0]
            if len(sudden_changes) > 5:  # More than 5 sudden changes
                anomalies.append('multiple_volume_changes')
            
            # Check for silence periods
            silence_threshold = np.percentile(rms, 10)
            silence_frames = np.sum(rms < silence_threshold)
            silence_ratio = silence_frames / len(rms)
            
            if silence_ratio > 0.3:  # More than 30% silence
                anomalies.append('excessive_silence')
            
            # Check for frequency anomalies
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_std = np.std(spectral_centroids)
            
            if spectral_std < 100:  # Very low frequency variation
                anomalies.append('low_frequency_variation')
            
            return {
                'anomalies': anomalies,
                'silence_ratio': silence_ratio,
                'volume_changes': len(sudden_changes),
                'frequency_variation': spectral_std
            }
            
        except Exception as e:
            print(f"Error detecting audio anomalies: {e}")
            return {
                'anomalies': ['analysis_error'],
                'silence_ratio': 0.0,
                'volume_changes': 0,
                'frequency_variation': 0.0
            }
    
    def analyze_audio_transcript(self, transcript):
        """Analyze transcribed text for misinformation."""
        if not transcript or len(transcript.strip()) < 10:
            return {
                'misinformation_score': 0.5,
                'confidence_score': 0.0,
                'llm_origin_probability': 0.5,
                'label': 'insufficient_content'
            }
        
        # Use text analysis for the transcript
        return analyze_text(transcript)
    
    def analyze_audio(self, audio_path):
        """Comprehensive audio analysis."""
        if not audio_path:
            return {
                'transcript': '',
                'transcript_confidence': 0.0,
                'misinformation_score': 0.5,
                'confidence_score': 0.0,
                'llm_origin_probability': 0.5,
                'audio_quality': {},
                'anomalies': [],
                'error': 'No audio path provided'
            }
        
        try:
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            if transcription.get('error'):
                return {
                    'transcript': '',
                    'transcript_confidence': 0.0,
                    'misinformation_score': 0.5,
                    'confidence_score': 0.0,
                    'llm_origin_probability': 0.5,
                    'audio_quality': {},
                    'anomalies': [],
                    'error': transcription['error']
                }
            
            # Analyze audio quality
            quality_metrics = self.analyze_audio_quality(audio_path)
            
            # Detect anomalies
            anomaly_detection = self.detect_audio_anomalies(audio_path)
            
            # Analyze transcript for misinformation
            transcript_analysis = self.analyze_audio_transcript(transcription['transcript'])
            
            return {
                'transcript': transcription['transcript'],
                'transcript_confidence': transcription['confidence'],
                'misinformation_score': transcript_analysis['misinformation_score'],
                'confidence_score': transcript_analysis['confidence_score'],
                'llm_origin_probability': transcript_analysis['llm_origin_probability'],
                'entropy_score': transcript_analysis.get('entropy_score', 0.0),
                'audio_quality': quality_metrics,
                'anomalies': anomaly_detection['anomalies'],
                'anomaly_metrics': {
                    'silence_ratio': anomaly_detection['silence_ratio'],
                    'volume_changes': anomaly_detection['volume_changes'],
                    'frequency_variation': anomaly_detection['frequency_variation']
                },
                'language': transcription.get('language', 'en'),
                'segments': transcription.get('segments', [])
            }
            
        except Exception as e:
            print(f"Error in comprehensive audio analysis: {e}")
            return {
                'transcript': '',
                'transcript_confidence': 0.0,
                'misinformation_score': 0.5,
                'confidence_score': 0.0,
                'llm_origin_probability': 0.5,
                'audio_quality': {},
                'anomalies': [],
                'error': str(e)
            }
    
    def analyze_multiple_audio_files(self, audio_paths):
        """Analyze multiple audio files."""
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            try:
                result = self.analyze_audio(audio_path)
                result['audio_index'] = i
                result['audio_path'] = audio_path
                results.append(result)
            except Exception as e:
                print(f"Error analyzing audio file {i}: {e}")
                results.append({
                    'transcript': '',
                    'transcript_confidence': 0.0,
                    'misinformation_score': 0.5,
                    'confidence_score': 0.0,
                    'llm_origin_probability': 0.5,
                    'audio_quality': {},
                    'anomalies': [],
                    'error': str(e),
                    'audio_index': i,
                    'audio_path': audio_path
                })
        
        return results

# Global analyzer instance
audio_analyzer = AudioAnalyzer()

def transcribe_audio(audio_path):
    """Convenience function to transcribe audio."""
    return audio_analyzer.transcribe_audio(audio_path)

def analyze_audio(audio_path):
    """Convenience function to analyze audio."""
    return audio_analyzer.analyze_audio(audio_path)

def analyze_multiple_audio_files(audio_paths):
    """Convenience function to analyze multiple audio files."""
    return audio_analyzer.analyze_multiple_audio_files(audio_paths) 