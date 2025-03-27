"""
Main Flask application for sentiment analysis web dashboard.
"""
import os
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from collections import defaultdict, deque

from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from src.data.real_time_collector import RealTimeDataCollector
from src.models.genai import GenAIAnalyzer
from src.models.traditional import LexiconAnalyzer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-placeholder')

# Initialize sentiment analyzers
genai_analyzer = GenAIAnalyzer()
traditional_analyzer = LexiconAnalyzer()

# Initialize data collector
data_collector = RealTimeDataCollector()

# In-memory data storage (replace with database for production)
MAX_ENTRIES = 1000
sentiment_data = deque(maxlen=MAX_ENTRIES)
source_stats = defaultdict(int)
sentiment_stats = {
    "positive": 0,
    "negative": 0,
    "neutral": 0
}
hourly_data = []

# Processing queue
data_queue = queue.Queue(maxsize=1000)
should_process = True

def configure_data_sources():
    """Configure and start data sources based on environment variables."""
    # Twitter configuration
    if os.getenv('TWITTER_BEARER_TOKEN'):
        from src.data.real_time_collector import TwitterStreamSource
        twitter_config = {
            "bearer_token": os.getenv('TWITTER_BEARER_TOKEN'),
            "api_key": os.getenv('TWITTER_API_KEY'),
            "api_secret": os.getenv('TWITTER_API_SECRET'),
            "access_token": os.getenv('TWITTER_ACCESS_TOKEN'),
            "access_secret": os.getenv('TWITTER_ACCESS_SECRET'),
            "rules": [
                {"value": "lang:en"}
            ]
        }
        data_collector.add_source(TwitterStreamSource(twitter_config))
    
    # News API configuration
    if os.getenv('NEWS_API_KEY'):
        from src.data.real_time_collector import NewsAPISource
        news_config = {
            "api_key": os.getenv('NEWS_API_KEY'),
            "polling_interval": 300,  # 5 minutes
            "language": "en"
        }
        data_collector.add_source(NewsAPISource(news_config))
    
    # Reddit configuration
    from src.data.real_time_collector import RedditDataSource
    reddit_config = {
        "polling_interval": 60,  # 1 minute
        "subreddits": ["news", "worldnews", "technology"],
        "limit": 25
    }
    data_collector.add_source(RedditDataSource(reddit_config))

def data_collection_thread():
    """Thread function to collect data from sources."""
    for item in data_collector.get_all_data():
        if not should_process:
            break
        try:
            data_queue.put(item, block=False)
        except queue.Full:
            pass  # Skip if queue is full

def data_processing_thread():
    """Thread function to process data in the queue."""
    batch = []
    last_process_time = time.time()
    
    while should_process:
        # Get item from queue with timeout
        try:
            item = data_queue.get(block=True, timeout=0.5)
            batch.append(item)
        except queue.Empty:
            pass
        
        current_time = time.time()
        # Process batch when it reaches size or time limit
        if (len(batch) >= 10 or 
                (batch and current_time - last_process_time > 2.0)):
            process_batch(batch)
            batch = []
            last_process_time = current_time

def process_batch(batch):
    """Process a batch of text data items."""
    if not batch:
        return
        
    # Extract text from items
    texts = [item.get('text', '') for item in batch]
    
    # Get sentiment from both analyzers
    try:
        genai_results = genai_analyzer.batch_predict(texts)
        traditional_results = traditional_analyzer.batch_predict(texts)
        
        # Combine results with original data
        for i, item in enumerate(batch):
            # Enhance with sentiment results
            item['analysis'] = {
                'genai': genai_results[i],
                'traditional': traditional_results[i],
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine consensus sentiment
            genai_sentiment = genai_results[i]['sentiment']
            trad_sentiment = traditional_results[i]['sentiment']
            
            if genai_sentiment == trad_sentiment:
                consensus = genai_sentiment
            elif 'neutral' in [genai_sentiment, trad_sentiment]:
                # If one is neutral, use the other
                consensus = genai_sentiment if trad_sentiment == 'neutral' else trad_sentiment
            else:
                # If they disagree (one positive, one negative), use one with higher confidence
                if genai_results[i]['confidence'] > traditional_results[i]['confidence']:
                    consensus = genai_sentiment
                else:
                    consensus = trad_sentiment
            
            item['analysis']['consensus'] = consensus
            
            # Update stats
            source = item.get('source', 'unknown')
            source_stats[source] += 1
            sentiment_stats[consensus] += 1
            
            # Add to storage
            sentiment_data.append(item)
        
        # Update hourly data
        update_hourly_data()
        
    except Exception as e:
        app.logger.error(f"Error processing batch: {e}")

def update_hourly_data():
    """Update hourly sentiment statistics."""
    global hourly_data
    
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    # Initialize if empty
    if not hourly_data:
        hourly_data = [
            {
                'hour': (current_hour - timedelta(hours=i)).isoformat(),
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'total': 0
            }
            for i in range(24)
        ]
        return
    
    # Check if we need to add a new hour
    latest_hour = datetime.fromisoformat(hourly_data[0]['hour'])
    if current_hour > latest_hour:
        # Add new hour
        hourly_data.insert(0, {
            'hour': current_hour.isoformat(),
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0
        })
        # Remove oldest hour
        if len(hourly_data) > 24:
            hourly_data.pop()
    
    # Update current hour stats
    current_hour_data = hourly_data[0]
    
    # Count recent sentiments (last minute)
    recent_cutoff = now - timedelta(minutes=1)
    recent_items = [
        item for item in sentiment_data
        if datetime.fromisoformat(item['analysis']['timestamp']) > recent_cutoff
    ]
    
    for item in recent_items:
        sentiment = item['analysis']['consensus']
        if sentiment in ['positive', 'negative', 'neutral']:
            current_hour_data[sentiment] += 1
            current_hour_data['total'] += 1

# Routes
@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get current stats."""
    return jsonify({
        'sources': dict(source_stats),
        'sentiments': sentiment_stats,
        'total': len(sentiment_data),
        'hourly': hourly_data
    })

@app.route('/api/recent')
def get_recent():
    """Get recent sentiment data."""
    limit = min(int(request.args.get('limit', 20)), 100)
    return jsonify([
        {
            'id': item.get('id', ''),
            'text': item.get('text', '')[:200] + ('...' if len(item.get('text', '')) > 200 else ''),
            'source': item.get('source', 'unknown'),
            'sentiment': item['analysis']['consensus'],
            'timestamp': item['analysis']['timestamp'],
            'confidence': max(
                item['analysis']['genai']['confidence'],
                item['analysis']['traditional']['confidence']
            )
        }
        for item in list(sentiment_data)[-limit:]
    ])

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze custom text."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    genai_result = genai_analyzer.predict(text)[0]
    trad_result = traditional_analyzer.predict(text)[0]
    
    return jsonify({
        'text': text,
        'genai': genai_result,
        'traditional': trad_result,
        'timestamp': datetime.now().isoformat()
    })

def start_background_threads():
    """Start background threads for data collection and processing."""
    global should_process
    should_process = True
    
    # Configure data sources
    configure_data_sources()
    
    # Start data collector
    data_collector.start_all()
    
    # Start threads
    collection_thread = threading.Thread(target=data_collection_thread)
    collection_thread.daemon = True
    collection_thread.start()
    
    processing_thread = threading.Thread(target=data_processing_thread)
    processing_thread.daemon = True
    processing_thread.start()
    
    app.logger.info("Background threads started")

def stop_background_threads():
    """Stop background threads."""
    global should_process
    should_process = False
    
    # Stop data collector
    data_collector.stop_all()
    
    app.logger.info("Background threads stopping")

@app.before_first_request
def before_first_request():
    """Initialize before first request."""
    start_background_threads()

@app.teardown_appcontext
def teardown_appcontext(exception=None):
    """Clean up resources."""
    pass

if __name__ == '__main__':
    # Start background threads
    start_background_threads()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
    finally:
        # Stop background threads
        stop_background_threads()