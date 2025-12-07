"""
AI Agent for Atomberg Share of Voice (SoV) Analysis
Analyzes YouTube search results to quantify brand presence and engagement
"""

import os
import json
import time
from datetime import datetime
from collections import defaultdict
import re

# Required installations:
# pip install google-api-python-client pandas matplotlib seaborn transformers torch

from googleapiclient.discovery import build
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline


# CONFIGURATION

YOUTUBE_API_KEY = "AIzaSyAIjDJnk9cgS6BBhYX8_w00fyirHlxedp8"  # Replace withrespective YouTube Data API key

# Keywords to analyze
KEYWORDS = [
    "smart fan",
    "smart fan India",
    "ceiling fan",
    "energy efficient fan",
    "best ceiling fan 2025",
    "Atomberg fan review"
]

# Number of results per keyword
N_RESULTS = 30 #Top30 results

# Comment analysis settings
ANALYZE_COMMENTS = True  # Set to False to skip comments (faster, saves quota)
MAX_COMMENTS_PER_VIDEO = 20  # Number of top comments to analyze
COMMENT_WEIGHT = 0.5  # Weight for comments vs video content (0.0 to 1.0)
                       # 0.5 = equal weight, 0.3 = less emphasis on comments

# Brand names to track
BRANDS = {
    "Atomberg": ["atomberg", "atom berg"],
    "Havells": ["havells"],
    "Orient": ["orient electric", "orient fan", "orient"],
    "Crompton": ["crompton", "crompton greaves"],
    "Usha": ["usha"],
    "Bajaj": ["bajaj"]
}

# YOUTUBE DATA COLLECTION

class YouTubeAnalyzer:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                          model="distilbert-base-uncased-finetuned-sst-2-english",
                                          framework="pt")
    
    def search_videos(self, keyword, max_results=30):
        """Search YouTube for videos matching keyword"""
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=keyword,
                type="video",
                maxResults=max_results,
                order="relevance",
                regionCode="IN"
            )
            response = request.execute()
            
            video_ids = [item['id']['videoId'] for item in response['items']]
            return self.get_video_details(video_ids)
        
        except Exception as e:
            print(f"Error searching for '{keyword}': {e}")
            return []
    
    def get_video_details(self, video_ids):
        """Get detailed statistics for videos"""
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=','.join(video_ids)
            )
            response = request.execute()
            
            videos = []
            for item in response['items']:
                video = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'comments': int(item['statistics'].get('commentCount', 0))
                }
                videos.append(video)
            
            return videos
        
        except Exception as e:
            print(f"Error getting video details: {e}")
            return []
    
    def get_top_comments(self, video_id, max_comments=20):
        """Get top comments from a video"""
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_comments,
                order="relevance"
            )
            response = request.execute()
            
            comments = []
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            
            return comments
        
        except Exception as e:
            return []

# BRAND DETECTION & SENTIMENT ANALYSIS

def detect_brands(text):
    """Detect which brands are mentioned in text"""
    text_lower = text.lower()
    mentioned_brands = []
    
    for brand, keywords in BRANDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                mentioned_brands.append(brand)
                break
    
    return list(set(mentioned_brands))

def analyze_sentiment(text, sentiment_analyzer):
    """Analyze sentiment of text"""
    try:
        # Truncate to 512 tokens for the model
        text = text[:512]
        result = sentiment_analyzer(text)[0]
        
        # Convert to score: POSITIVE=1, NEGATIVE=-1
        score = 1 if result['label'] == 'POSITIVE' else -1
        confidence = result['score']
        
        return score * confidence
    except:
        return 0

def analyze_brand_specific_sentiment(text, brand, sentiment_analyzer):
    """
    Analyze sentiment specifically for a brand by extracting context around brand mentions.
    This handles cases where a video mentions multiple brands with different sentiments.
    """
    text_lower = text.lower()
    brand_keywords = BRANDS[brand]
    
    # Find all sentences containing the brand
    sentences = re.split(r'[.!?\n]+', text)
    brand_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for keyword in brand_keywords:
            if keyword in sentence_lower:
                brand_sentences.append(sentence.strip())
                break
    
    # If no specific mentions, fall back to general sentiment
    if not brand_sentences:
        return analyze_sentiment(text, sentiment_analyzer)
    
    # Analyze sentiment of brand-specific context
    brand_context = ' '.join(brand_sentences)
    
    context_window = 100  # number characters before and after brand mention
    contexts = []
    
    for keyword in brand_keywords:
        pos = 0
        while True:
            pos = text_lower.find(keyword, pos)
            if pos == -1:
                break
            
            start = max(0, pos - context_window)
            end = min(len(text), pos + len(keyword) + context_window)
            context = text[start:end]
            contexts.append(context)
            pos += len(keyword)
    
    # Combine brand sentences and context windows
    combined_text = brand_context + ' ' + ' '.join(contexts)
    
    return analyze_sentiment(combined_text, sentiment_analyzer)

# SHARE OF VOICE CALCULATION

def calculate_engagement_score(video):
    """Calculate weighted engagement score"""
    # Weights: views + likes*10 + comments*20
    return video['views'] + (video['likes'] * 10) + (video['comments'] * 20)

def calculate_sov(all_videos, analyzer):
    """Calculate Share of Voice metrics for all brands"""
    
    brand_data = defaultdict(lambda: {
        'mentions': 0,
        'total_engagement': 0,
        'positive_sentiment': 0,
        'negative_sentiment': 0,
        'videos': [],
        'comment_sentiment': 0,  
        'video_sentiment': 0     
    })
    
    total_videos = len(all_videos)
    total_engagement = 0
    
    for video in all_videos:
        # Combine title and description for analysis
        content = f"{video['title']} {video['description']}"
        brands_mentioned = detect_brands(content)
        
        if not brands_mentioned:
            continue
        
        engagement = calculate_engagement_score(video)
        total_engagement += engagement
        
        # Get comments for more context
        comments = []
        if ANALYZE_COMMENTS:
            try:
                print(f"      Fetching comments for: {video['title'][:50]}...")
                comments = analyzer.get_top_comments(video['video_id'], max_comments=MAX_COMMENTS_PER_VIDEO)
                print(f"      Retrieved {len(comments)} comments")
            except Exception as e:
                print(f"      Could not fetch comments: {str(e)[:50]}")
                pass
        
        comments_text = ' '.join(comments)
        
        # Analyze sentiment for EACH brand separately
        for brand in brands_mentioned:
            brand_data[brand]['mentions'] += 1
            brand_data[brand]['total_engagement'] += engagement
            brand_data[brand]['videos'].append(video['video_id'])
            
            # Analyze video content sentiment (keep sign!)
            video_sentiment = analyze_brand_specific_sentiment(
                content, brand, analyzer.sentiment_analyzer
            )
            brand_data[brand]['video_sentiment'] += video_sentiment
            
            # Analyze comment sentiment separately (keep sign!)
            comment_sentiment = 0
            if comments_text:
                comment_sentiment = analyze_brand_specific_sentiment(
                    comments_text, brand, analyzer.sentiment_analyzer
                )
                brand_data[brand]['comment_sentiment'] += comment_sentiment
            
            # Combine video and comment sentiment with weighting
            if ANALYZE_COMMENTS and comments_text:
                combined_sentiment = (
                    video_sentiment * (1 - COMMENT_WEIGHT) + 
                    comment_sentiment * COMMENT_WEIGHT
                )
            else:
                combined_sentiment = video_sentiment
            
            # Now properly separate positive and negative
            if combined_sentiment > 0:
                brand_data[brand]['positive_sentiment'] += combined_sentiment
            else:
                brand_data[brand]['negative_sentiment'] += abs(combined_sentiment)
    
    # Calculate SoV percentages
    sov_results = {}
    
    for brand, data in brand_data.items():
        mention_share = (data['mentions'] / total_videos * 100) if total_videos > 0 else 0
        engagement_share = (data['total_engagement'] / total_engagement * 100) if total_engagement > 0 else 0
        
        total_sentiment = data['positive_sentiment'] + data['negative_sentiment']
        positive_share = (data['positive_sentiment'] / total_sentiment * 100) if total_sentiment > 0 else 0
        
        # Composite SoV Score: weighted average
        composite_sov = (mention_share * 0.3) + (engagement_share * 0.4) + (positive_share * 0.3)
        
        sov_results[brand] = {
            'mention_share': round(mention_share, 2),
            'engagement_share': round(engagement_share, 2),
            'positive_sentiment_share': round(positive_share, 2),
            'composite_sov': round(composite_sov, 2),
            'total_mentions': data['mentions'],
            'total_engagement': data['total_engagement']
        }
    
    return sov_results

# VISUALIZATION AND AGGREGATION

def create_visualizations(keyword_results):
    """Create charts for SoV analysis"""
    
    # Aggregate data across all keywords
    aggregated = defaultdict(lambda: {
        'composite_sov': 0,
        'mention_share': 0,
        'engagement_share': 0,
        'positive_sentiment_share': 0,
        'count': 0
    })
    
    for keyword, results in keyword_results.items():
        for brand, metrics in results.items():
            aggregated[brand]['composite_sov'] += metrics['composite_sov']
            aggregated[brand]['mention_share'] += metrics['mention_share']
            aggregated[brand]['engagement_share'] += metrics['engagement_share']
            aggregated[brand]['positive_sentiment_share'] += metrics['positive_sentiment_share']
            aggregated[brand]['count'] += 1
    
    # Average the scores
    for brand in aggregated:
        count = aggregated[brand]['count']
        if count > 0:
            aggregated[brand] = {k: v/count for k, v in aggregated[brand].items() if k != 'count'}
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Atomberg Share of Voice Analysis', fontsize=16, fontweight='bold')
    
    brands = list(aggregated.keys())
    composite_scores = [aggregated[b]['composite_sov'] for b in brands]
    
    # 1. Composite SoV Comparison
    axes[0, 0].barh(brands, composite_scores, color=['#FF6B6B' if b == 'Atomberg' else '#4ECDC4' for b in brands])
    axes[0, 0].set_xlabel('Composite SoV Score (%)')
    axes[0, 0].set_title('Composite Share of Voice')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Mention vs Engagement Share
    mention_shares = [aggregated[b]['mention_share'] for b in brands]
    engagement_shares = [aggregated[b]['engagement_share'] for b in brands]
    
    x = range(len(brands))
    width = 0.35
    axes[0, 1].bar([i - width/2 for i in x], mention_shares, width, label='Mention Share', alpha=0.8)
    axes[0, 1].bar([i + width/2 for i in x], engagement_shares, width, label='Engagement Share', alpha=0.8)
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].set_title('Mention Share vs Engagement Share')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(brands, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Positive Sentiment Share
    sentiment_shares = [aggregated[b]['positive_sentiment_share'] for b in brands]
    axes[1, 0].bar(brands, sentiment_shares, color=['#95E1D3' if b == 'Atomberg' else '#F38181' for b in brands])
    axes[1, 0].set_ylabel('Positive Sentiment (%)')
    axes[1, 0].set_title('Share of Positive Voice')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Keyword-wise Atomberg Performance
    atomberg_by_keyword = {k: v.get('Atomberg', {}).get('composite_sov', 0) 
                          for k, v in keyword_results.items()}
    
    keywords = list(atomberg_by_keyword.keys())
    scores = list(atomberg_by_keyword.values())
    
    axes[1, 1].barh(keywords, scores, color='#FF6B6B')
    axes[1, 1].set_xlabel('Composite SoV Score (%)')
    axes[1, 1].set_title('Atomberg Performance by Keyword')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sov_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved as 'sov_analysis_results.png'")
    
    return aggregated

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("ATOMBERG SHARE OF VOICE (SoV) ANALYSIS")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = YouTubeAnalyzer(YOUTUBE_API_KEY)
    
    all_keyword_results = {}
    all_videos = []
    
    # Analyze each keyword
    for keyword in KEYWORDS:
        print(f"\n🔍 Analyzing keyword: '{keyword}'")
        print("-" * 70)
        
        videos = analyzer.search_videos(keyword, N_RESULTS)
        all_videos.extend(videos)
        
        print(f"   Found {len(videos)} videos")
        
        # Calculate SoV for this keyword
        sov_results = calculate_sov(videos, analyzer)
        all_keyword_results[keyword] = sov_results
        
        # Display results
        if sov_results:
            print(f"\n   Share of Voice Results for '{keyword}':")
            for brand, metrics in sorted(sov_results.items(), 
                                        key=lambda x: x[1]['composite_sov'], 
                                        reverse=True):
                print(f"   • {brand:15} - SoV: {metrics['composite_sov']:5.2f}% "
                      f"(Mentions: {metrics['total_mentions']}, "
                      f"Engagement: {metrics['engagement_share']:.2f}%)")
        
        time.sleep(1)  # Rate limiting
    
    # Creating visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 70)
    
    aggregated_results = create_visualizations(all_keyword_results)
    
    # Saving detailed results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'keywords_analyzed': KEYWORDS,
        'total_videos_analyzed': len(all_videos),
        'keyword_results': all_keyword_results,
        'aggregated_results': dict(aggregated_results)
    }
    
    with open('sov_detailed_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Detailed results saved to 'sov_detailed_results.json'")
    
    print("\n" + "=" * 70)
    print("SUMMARY - AGGREGATED SHARE OF VOICE")
    print("=" * 70)
    
    for brand, metrics in sorted(aggregated_results.items(), 
                                key=lambda x: x[1]['composite_sov'], 
                                reverse=True):
        print(f"\n{brand}:")
        print(f"  Composite SoV:        {metrics['composite_sov']:.2f}%")
        print(f"  Mention Share:        {metrics['mention_share']:.2f}%")
        print(f"  Engagement Share:     {metrics['engagement_share']:.2f}%")
        print(f"  Positive Sentiment:   {metrics['positive_sentiment_share']:.2f}%")
    
    if ANALYZE_COMMENTS:
        print("\n" + "=" * 70)
        print("COMMENT SENTIMENT ANALYSIS")
        print(f"Settings: {MAX_COMMENTS_PER_VIDEO} comments per video, {COMMENT_WEIGHT*100}% weight")
        print("=" * 70)
        print("\nNote: Comment sentiment analyzed separately from video content")
        print("This provides insight into audience perception vs creator messaging")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()