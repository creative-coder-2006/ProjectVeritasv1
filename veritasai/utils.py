import os
import shutil
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text."""
    if not text:
        return 0.0
    
    # Count character frequencies
    char_freq = {}
    total_chars = len(text)
    
    for char in text.lower():
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    for freq in char_freq.values():
        if freq > 0:
            p = freq / total_chars
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_confidence_interval(scores: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate confidence interval for a list of scores."""
    if not scores:
        return {'mean': 0.0, 'std': 0.0, 'lower': 0.0, 'upper': 0.0}
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    
    # Calculate confidence interval
    n = len(scores)
    if n > 1:
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_of_error = t_value * std / np.sqrt(n)
    else:
        margin_of_error = 0.0
    
    return {
        'mean': mean,
        'std': std,
        'lower': max(0.0, mean - margin_of_error),
        'upper': min(1.0, mean + margin_of_error)
    }

def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {file_path}: {e}")

def cleanup_temp_directories(directory_paths: List[str]) -> None:
    """Clean up temporary directories."""
    for directory_path in directory_paths:
        try:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
                logger.info(f"Cleaned up directory: {directory_path}")
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory_path}: {e}")

def create_temp_directory(base_path: str, prefix: str = "veritas_") -> str:
    """Create a temporary directory."""
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=base_path)
    logger.info(f"Created temp directory: {temp_dir}")
    return temp_dir

def validate_file_path(file_path: str) -> bool:
    """Validate if a file path exists and is accessible."""
    return os.path.exists(file_path) and os.path.isfile(file_path)

def validate_directory_path(directory_path: str) -> bool:
    """Validate if a directory path exists and is accessible."""
    return os.path.exists(directory_path) and os.path.isdir(directory_path)

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0.0

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    import re
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:200-len(ext)] + ext
    return sanitized

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using Jaccard similarity."""
    if not text1 or not text2:
        return 0.0
    
    # Tokenize texts
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    if not text:
        return []
    
    import re
    from collections import Counter
    
    # Remove punctuation and convert to lowercase
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Return top keywords
    return [word for word, count in word_counts.most_common(max_keywords)]

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a score to the range [0, 1]."""
    if max_val == min_val:
        return 0.5
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def calculate_weighted_average(scores: List[float], weights: List[float]) -> float:
    """Calculate weighted average of scores."""
    if not scores or not weights or len(scores) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    return weighted_sum / total_weight

def log_analysis_step(step_name: str, details: Dict[str, Any] = None) -> None:
    """Log an analysis step for debugging and monitoring."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {step_name}"
    
    if details:
        log_message += f" - {details}"
    
    logger.info(log_message)

def create_progress_tracker(total_items: int, description: str = "Processing") -> callable:
    """Create a progress tracking function."""
    from tqdm import tqdm
    
    pbar = tqdm(total=total_items, desc=description)
    
    def update_progress(completed: int = 1):
        pbar.update(completed)
    
    def close_progress():
        pbar.close()
    
    update_progress.close = close_progress
    return update_progress

def validate_url(url: str) -> bool:
    """Validate URL format."""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))

def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_percentage(value: float, total: float) -> float:
    """Calculate percentage."""
    if total == 0:
        return 0.0
    return (value / total) * 100

def is_valid_json(data: str) -> bool:
    """Check if string is valid JSON."""
    try:
        import json
        json.loads(data)
        return True
    except:
        return False

def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string."""
    import json
    try:
        return json.dumps(obj, default=str)
    except:
        return str(obj) 