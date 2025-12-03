#!/usr/bin/env python3
'''
Trump Truth RSS Feed Parser
Fetches posts from https://trumpstruth.org/feed and stores them in a local database.
Sends Telegram notifications for new posts.
'''

import asyncio
import os
import logging
from datetime import datetime
import datetime as dt
from typing import List, Dict

import pytz
from tinydb import TinyDB, Query
from telegram import Bot, InputMediaPhoto
from telegram.request import HTTPXRequest
from telegram.error import RetryAfter
import re
import requests
from bs4 import BeautifulSoup

from feed_fetcher import safe_fetch_feed
from llm_summarize_v4_finance import TrumpFeedAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.getenv('LOG_PATH', 'trump_feed_parser.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DB_FILE = os.getenv('DB_PATH', 'trump_posts_db.json')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')

# Database setup
db = TinyDB(DB_FILE)
Post = Query()


def format_summary_for_telegram(summary: str) -> str:
    '''Convert feed summary HTML to Telegram-friendly format.
    Extracts and formats links, removes HTML tags, and cleans up the text.'''
    try:
        # Parse HTML
        soup = BeautifulSoup(summary, 'html.parser')

        # Find all links
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href:
                # Replace the link with a Telegram-friendly format
                link.replace_with(f'<a href="{href}">{href}</a>')

        # Get clean text without other HTML tags
        clean_text = soup.get_text().strip()

        # Remove multiple spaces and newlines
        clean_text = re.sub(r'\s+', ' ', clean_text)

        return clean_text

    except Exception as e:
        logger.error(f'Error formatting summary for Telegram: {e}')
        # Return a cleaned version of the original text as fallback
        return re.sub(r'<[^>]+>', '', summary).strip()


async def send_telegram_notification(post: dict, analysis: dict) -> bool:
    '''Send Telegram notification about new post.'''
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning('Telegram credentials not configured')
        return False

    # Configure request with longer timeouts for image uploads
    t_request = HTTPXRequest(connection_pool_size=8, connect_timeout=60.0, read_timeout=60.0, write_timeout=60.0)
    bot = Bot(token=TELEGRAM_TOKEN, request=t_request)

    # Extract analysis data
    headline = analysis.get('headline', 'New Post')
    summary_text = analysis.get('summary', '')
    relevance_score = analysis.get('relevance_score', 0)
    is_relevant = analysis.get('is_financial_relevant', False)
    market_impact = analysis.get('market_impact') or {}

    # Determine Header Emoji/Prefix
    if relevance_score >= 8:
        header = f"🚨 <b>MARKET MOVER: {headline}</b>"
    elif relevance_score >= 4:
        header = f"⚠️ <b>MARKET RELEVANT: {headline}</b>"
    else:
        header = f"📢 <b>{headline}</b>"

    # Build Message Body
    message_parts = [header, "", f"{summary_text}", ""]

    # Add Financial Impact Section if relevant
    if is_relevant and relevance_score > 0:
        sentiment = market_impact.get('sentiment', 'NEUTRAL')
        sentiment_emoji = "🐂" if sentiment == "BULLISH" else "🐻" if sentiment == "BEARISH" else "😐"

        signal = market_impact.get('signal', 'N/A')
        tickers = market_impact.get('tickers', [])
        sectors = market_impact.get('sectors', [])

        impact_section = [
            "📉 <b>Market Impact</b>",
            f"<b>Signal:</b> {signal}",
            f"<b>Sentiment:</b> {sentiment} {sentiment_emoji}",
            f"<b>Score:</b> {relevance_score}/10"
        ]

        if tickers:
            impact_section.append(f"<b>Tickers:</b> {', '.join(tickers)}")
        if sectors:
            impact_section.append(f"<b>Sectors:</b> {', '.join(sectors)}")

        message_parts.extend(impact_section)
        message_parts.append("") # Spacer

    # Footer
    ts_url = post.get('truth_social_url', post['link'])
    likes = post.get('likes', 0)
    retruths = post.get('retruths', 0)
    
    stats_line = ""
    if likes > 0 or retruths > 0:
        stats_line = f"❤️ {likes:,}  🔁 {retruths:,}\n"

    footer = (
        f'{stats_line}'
        f'Published: {to_hr_format(post["published"])}\n'
        f'<a href="{ts_url}">Read original post</a>'
    )
    message_parts.append(footer)

    final_message = "\n".join(message_parts)

    # Handle Images
    images = post.get('images', [])

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with bot:
                # If we have images, send them
                if images:
                    # Check if caption fits
                    caption_fits = len(final_message) <= 1024
                    
                    if len(images) == 1:
                        # Single Image
                        if caption_fits:
                            await bot.send_photo(
                                chat_id=TELEGRAM_CHAT_ID,
                                photo=images[0],
                                caption=final_message,
                                parse_mode='HTML'
                            )
                            if TELEGRAM_CHANNEL_ID:
                                await bot.send_photo(
                                    chat_id=TELEGRAM_CHANNEL_ID,
                                    photo=images[0],
                                    caption=final_message,
                                    parse_mode='HTML'
                                )
                        else:
                            # Send photo then text
                            await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=images[0])
                            await bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=final_message,
                                parse_mode='HTML',
                                disable_web_page_preview=True,
                            )
                            if TELEGRAM_CHANNEL_ID:
                                await bot.send_photo(chat_id=TELEGRAM_CHANNEL_ID, photo=images[0])
                                await bot.send_message(
                                    chat_id=TELEGRAM_CHANNEL_ID,
                                    text=final_message,
                                    parse_mode='HTML',
                                    disable_web_page_preview=True,
                                )
                    else:
                        # Multiple Images (Media Group)
                        # Telegram limits media groups to 10 items
                        media_group = []
                        for idx, img_url in enumerate(images[:10]):
                            if idx == 0 and caption_fits:
                                media_group.append(InputMediaPhoto(media=img_url, caption=final_message, parse_mode='HTML'))
                            else:
                                media_group.append(InputMediaPhoto(media=img_url))
                        
                        await bot.send_media_group(chat_id=TELEGRAM_CHAT_ID, media=media_group)
                        if TELEGRAM_CHANNEL_ID:
                            # Re-create media group for the second send to avoid any potential issues with reused input objects
                            media_group_channel = []
                            for idx, img_url in enumerate(images[:10]):
                                if idx == 0 and caption_fits:
                                    media_group_channel.append(InputMediaPhoto(media=img_url, caption=final_message, parse_mode='HTML'))
                                else:
                                    media_group_channel.append(InputMediaPhoto(media=img_url))
                            await bot.send_media_group(chat_id=TELEGRAM_CHANNEL_ID, media=media_group_channel)
                            
                        # If caption didn't fit, send it as a separate message
                        if not caption_fits:
                            await bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=final_message,
                                parse_mode='HTML',
                                disable_web_page_preview=True,
                            )
                            if TELEGRAM_CHANNEL_ID:
                                await bot.send_message(
                                    chat_id=TELEGRAM_CHANNEL_ID,
                                    text=final_message,
                                    parse_mode='HTML',
                                    disable_web_page_preview=True,
                                )
                else:
                    # Send text only
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=final_message,
                        parse_mode='HTML',
                        disable_web_page_preview=True,
                    )

                    # Send to channel
                    if TELEGRAM_CHANNEL_ID:
                        await bot.send_message(
                            chat_id=TELEGRAM_CHANNEL_ID,
                            text=final_message,
                            parse_mode='HTML',
                            disable_web_page_preview=True,
                        )
            return True
            
        except RetryAfter as e:
            retry_count += 1
            wait_time = e.retry_after + 1  # Add 1 second buffer
            logger.warning(f"Telegram Flood Control: Sleeping for {wait_time} seconds. Retry {retry_count}/{max_retries}")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
            
    logger.error(f"Failed to send Telegram notification after {max_retries} retries")
    return False


def get_truth_social_data(trumpstruth_url: str) -> Dict:
    """
    Fetches additional data from Truth Social via trumpstruth.org link.
    Returns a dict with: text, images (list), likes (int), retruths (int).
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    result = {
        'text': None,
        'images': [],
        'likes': 0,
        'retruths': 0,
        'truth_social_url': None
    }
    
    try:
        # 1. Get the Truth Social Link from Trumpstruth.org
        resp = requests.get(trumpstruth_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Failed to fetch {trumpstruth_url}: {resp.status_code}")
            return result
            
        soup = BeautifulSoup(resp.content, 'html.parser')
        external_link = soup.find('a', class_='status__external-link')
        
        if not external_link:
            logger.warning(f"No external link found on {trumpstruth_url}")
            return result
            
        truth_social_url = external_link.get('href')
        result['truth_social_url'] = truth_social_url
        
        # 2. Extract ID and call API
        # URL format: https://truthsocial.com/@realDonaldTrump/115656343773820545
        match = re.search(r'/(\d+)$', truth_social_url)
        if not match:
            logger.warning(f"Could not extract ID from {truth_social_url}")
            return result
            
        post_id = match.group(1)
        api_url = f"https://truthsocial.com/api/v1/statuses/{post_id}"
        
        api_resp = requests.get(api_url, headers=headers, timeout=10)
        if api_resp.status_code != 200:
            logger.warning(f"Failed to fetch API {api_url}: {api_resp.status_code}")
            return result
            
        data = api_resp.json()
        
        # 3. Parse Data
        result['likes'] = data.get('favourites_count', 0)
        result['retruths'] = data.get('reblogs_count', 0)
        
        # Clean HTML content for text
        raw_content = data.get('content', '')
        if raw_content:
            result['text'] = format_summary_for_telegram(raw_content)
            
        # Extract Images
        media_attachments = data.get('media_attachments', [])
        for media in media_attachments:
            if media.get('type') == 'image' and media.get('url'):
                result['images'].append(media.get('url'))
                
    except Exception as e:
        logger.error(f"Error fetching Truth Social data: {e}")
        
    return result


def fetch_feed(url: str) -> List[Dict]:
    # Using safe fetch method (returns None on failure)
    entries = safe_fetch_feed(url)

    return entries


def parse_published_date(date_str: str) -> str:
    # date_str, e.g.: 'Sat, 26 Apr 2025 02:19:29 +0000'
    input_format = '%a, %d %b %Y %H:%M:%S %z'
    dt_utc = datetime.strptime(date_str, input_format)
    dt_iso_format = dt_utc.isoformat()

    return dt_iso_format


def format_date_to_msk_tz(date_str: str) -> str:
    moscow_tz = pytz.timezone('Europe/Moscow')
    dt_moscow = datetime.fromisoformat(date_str).astimezone(moscow_tz)
    dt_iso_format = dt_moscow.isoformat()
    
    return dt_iso_format


def to_hr_format(date_str: str) -> str:
    dt_obj = datetime.fromisoformat(date_str)
    dt_hr_format = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
    
    return dt_hr_format


def format_post_entry(entry: dict) -> dict:
    '''Format a single post entry from the RSS feed.'''
    return dict(
        title=entry['title'],
        summary=entry['summary'],
        link=entry['link'],
        id=entry['id'].split('/')[-1],
        published=format_date_to_msk_tz(parse_published_date(entry['published'])),
        added_at = format_date_to_msk_tz(datetime.now(dt.timezone.utc).isoformat(timespec='seconds'))
    )


async def process_new_posts(entries: List[Dict], analyzer: TrumpFeedAnalyzer) -> int:
    '''Process new posts and store in database, returns count of new posts added.
    Processes entries in reverse chronological order (oldest first) to maintain proper sequence.'''
    
    # Sort entries by published date in ascending order (oldest first)
    sorted_entries = sorted(entries, key=lambda x: parse_published_date(x['published']))
    
    new_posts_to_process = []
    
    # Identify new posts
    for entry in sorted_entries:
        entry_dict = format_post_entry(entry)
        if not db.contains(Post.id == entry_dict['id']):
            new_posts_to_process.append(entry_dict)

    if not new_posts_to_process:
        logger.info('No new posts found.')
        return 0

    logger.info(f'Found {len(new_posts_to_process)} new posts. Starting batch analysis...')

    # Prepare texts for analysis (extract cleaned summary)
    texts_to_analyze = [format_summary_for_telegram(p['summary']) for p in new_posts_to_process]
    
    try:
        # Batch analyze
        analysis_results_list = await analyzer.analyze_batch(texts_to_analyze)
        # Convert list to a dictionary keyed by post_id for safe lookup
        analysis_map = {res.get('post_id'): res for res in analysis_results_list}
    except Exception as e:
        logger.error(f"Critical error during batch analysis: {e}")
        analysis_map = {}

    # Process results and send notifications
    processed_count = 0
    
    for i, post in enumerate(new_posts_to_process):
        # Enrich post with Truth Social data (Images, Stats, Full Text)
        ts_data = get_truth_social_data(post['link'])
        
        # Update post with enriched data
        if ts_data['text']:
            post['full_text'] = ts_data['text']
        if ts_data['images']:
            post['images'] = ts_data['images']
        post['likes'] = ts_data['likes']
        post['retruths'] = ts_data['retruths']
        post['truth_social_url'] = ts_data['truth_social_url'] or post['link']

        # Safe Lookup using the index 'i' which corresponds to 'post_id'
        analysis = analysis_map.get(i)
        
        if not analysis:
            logger.error(f"Analysis result missing for post index {i} (ID: {post['id']})")
            # Use enriched text if available, else summary
            text_for_summary = post.get('full_text') or format_summary_for_telegram(post['summary'])
            analysis = {
                "headline": "New Post (Analysis Failed)",
                "summary": text_for_summary,
                "is_financial_relevant": False,
                "relevance_score": 0
            }

        # Save to DB
        db.insert(post)
        logger.info(f"Processing new post, id:{post['id']}")
        
        # Send Notification
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            await send_telegram_notification(post, analysis)
        
        processed_count += 1
        
        # Rate limiting buffer: Sleep 2 seconds between posts to avoid hitting Telegram limits
        await asyncio.sleep(2)

    return processed_count


async def main():
    FEED_URL = 'https://trumpstruth.org/feed'
    POLL_INTERVAL = 60 * 5  # 5 minutes in seconds
    
    # Initialize Analyzer once
    try:
        analyzer = TrumpFeedAnalyzer()
        logger.info('TrumpFeedAnalyzer initialized successfully.')
    except Exception as e:
        logger.critical(f"Failed to initialize TrumpFeedAnalyzer: {e}")
        return

    logger.info('Trump Truth RSS Feed Parser started')
    while True:
        try:
            logger.info('Fetching feed...')
            posts = fetch_feed(FEED_URL)
            logger.info(f'Fetched {len(posts)} posts.')
            new_posts = await process_new_posts(posts, analyzer)
            logger.info(f'Processed {new_posts} new posts out of {len(posts)} total posts.')
        except Exception as e:
            logger.error(f'Error processing feed: {e}')
        
        await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    asyncio.run(main())
