import feedparser
import logging
import requests
import time
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)


def fetch_feed(url: str, timeout: int = 30, max_retries: int = 3, retry_delay: float = 2.0) -> List[Dict]:
    '''
    Fetch and parse RSS feed from given URL with timeout and retry logic.
    
    Args:
        url: RSS feed URL
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 2.0)
    
    Returns:
        List of feed entries
        
    Raises:
        Exception: If all retry attempts fail
    '''
    last_error = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            logger.info(f'Fetching RSS feed (attempt {attempt + 1}/{max_retries + 1})...')
            
            # Method 1: Use requests with timeout, then feedparser
            # This gives us better control over timeouts
            response = requests.get(
                url, 
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            response.raise_for_status()  # Raises exception for bad status codes
            
            # Parse the content with feedparser
            feed = feedparser.parse(response.content)
            
            # Check if parsing was successful
            if hasattr(feed, 'bozo') and feed.bozo == 1 and feed.entries == []:
                raise Exception(f'Failed to parse RSS feed: {getattr(feed, 'bozo_exception', 'Unknown parsing error')}')
            
            logger.info(f'Successfully fetched {len(feed.entries)} entries')
            return feed.entries
            
        except requests.exceptions.Timeout as e:
            last_error = f'Timeout error: {e}'
            logger.error(f'Attempt {attempt + 1} failed: {last_error}')
            
        except requests.exceptions.ConnectionError as e:
            last_error = f'Connection error: {e}'
            logger.error(f'Attempt {attempt + 1} failed: {last_error}')
            
        except requests.exceptions.HTTPError as e:
            last_error = f'HTTP error: {e}'
            logger.error(f'Attempt {attempt + 1} failed: {last_error}')
            
        except requests.exceptions.RequestException as e:
            last_error = f'Request error: {e}'
            logger.error(f'Attempt {attempt + 1} failed: {last_error}')
            
        except Exception as e:
            last_error = f'Unexpected error: {e}'
            logger.error(f'Attempt {attempt + 1} failed: {last_error}')
        
        # Don't delay after the last attempt
        if attempt < max_retries:
            logger.info(f'Retrying in {retry_delay} seconds...')
            time.sleep(retry_delay)
            # Exponential backoff: increase delay for next retry
            retry_delay *= 1.5
    
    # If we get here, all attempts failed
    raise Exception(f'Failed to fetch RSS feed after {max_retries + 1} attempts. Last error: {last_error}')


def safe_fetch_feed(url: str, timeout: int = 30, max_retries: int = 3) -> Optional[List[Dict]]:
    '''
    Safe wrapper that returns None instead of raising exceptions.
    Useful when you want to continue processing even if feed fetch fails.
    
    Args:
        url: RSS feed URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of feed entries or None if failed
    '''
    try:
        return fetch_feed(url, timeout=timeout, max_retries=max_retries)
    except Exception as e:
        logger.error(f'Feed fetch failed completely: {e}')
        return None


# Example usage
if __name__ == '__main__':
    # Trump's Truth Social RSS (example URL - replace with actual)
    rss_url = 'https://trumpstruth.org/feed'
    
    try:
        # Method 1: With exceptions (recommended)
        entries = fetch_feed(rss_url, timeout=15, max_retries=2)
        logger.info(f'Fetched {len(entries)} blog posts')
        
        # Print first entry as example
        if entries:
            first_entry = entries[0]
            logger.info(f'\nFirst entry: {first_entry.get('title', 'No title')}')
            
    except Exception as e:
        print(f'Could not fetch RSS feed: {e}')
        
        # Method 2: Try fallback method
        # try:
        #     entries = fetch_feed_fallback(rss_url, timeout=15, max_retries=2)
        #     logger.info(f'Fallback method succeeded: {len(entries)} entries')
        # except Exception as e2:
        #     logger.error(f'Fallback also failed: {e2}')
    
    # Method 3: Safe method (returns None on failure)
    entries = safe_fetch_feed(rss_url, timeout=10)
    if entries:
        logger.info(f'Safe method got {len(entries)} entries')
    else:
        logger.warning('Safe method returned None')