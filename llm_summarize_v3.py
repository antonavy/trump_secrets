from google import genai
from google.genai import types
import asyncio
from collections import deque
import os
import re
import time

class GeminiSummarizer:
    def __init__(self, api_key: str | None = None, model_name: str = 'gemini-2.0-flash', max_rpm: int = 15):
        '''
        Initialize the Gemini summarizer.
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from GEMINI_API_KEY env var.
            model_name: Gemini model to use (default: gemini-2.0-flash).
            max_rpm: Maximum requests per minute for rate limiting (default: 15 for free tier).
        '''
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError('GEMINI_API_KEY must be provided or set in environment')
        
        # Initialize client with API key
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        # Rate limiting configuration
        self.max_rpm = max_rpm
        self._request_timestamps: deque = deque()
        self._rate_limit_lock = asyncio.Lock()
        
        # Safety settings to prevent content blocking for news/political content
        self.safety_settings = [
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE',
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE',
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_NONE',
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE',
            ),
        ]
        
        # Generation config for controlled output
        self.generation_config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.9,
            max_output_tokens=100,
            safety_settings=self.safety_settings,
        )
    
    def _wait_for_rate_limit(self) -> None:
        '''
        Wait if necessary to stay within rate limits.
        Uses a sliding window to track requests in the last 60 seconds.
        '''
        current_time = time.time()
        
        # Remove timestamps older than 60 seconds
        while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
            self._request_timestamps.popleft()
        
        # If at rate limit, wait until oldest request expires
        if len(self._request_timestamps) >= self.max_rpm:
            sleep_time = 60 - (current_time - self._request_timestamps[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Clean up after sleeping
            current_time = time.time()
            while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
                self._request_timestamps.popleft()
        
        # Record this request
        self._request_timestamps.append(time.time())
    
    async def _async_wait_for_rate_limit(self) -> None:
        '''
        Async version: Wait if necessary to stay within rate limits.
        Uses a sliding window to track requests in the last 60 seconds.
        '''
        async with self._rate_limit_lock:
            current_time = time.time()
            
            # Remove timestamps older than 60 seconds
            while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
                self._request_timestamps.popleft()
            
            # If at rate limit, wait until oldest request expires
            if len(self._request_timestamps) >= self.max_rpm:
                sleep_time = 60 - (current_time - self._request_timestamps[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                # Clean up after sleeping
                current_time = time.time()
                while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
                    self._request_timestamps.popleft()
            
            # Record this request
            self._request_timestamps.append(time.time())
    
    def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig, max_retries: int = 3):
        '''
        Call client.models.generate_content with retry logic and rate limiting.
        
        Args:
            prompt: The prompt to send to the model
            config: Generation configuration (types.GenerateContentConfig)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Response from the model
            
        Raises:
            Exception: If all retries fail
        '''
        last_exception: Exception | None = None
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
        
        raise last_exception  # type: ignore[misc]
    
    async def _async_call_with_retry(self, prompt: str, config: types.GenerateContentConfig, max_retries: int = 3):
        '''
        Async version: Call client.aio.models.generate_content with retry logic and rate limiting.
        
        Args:
            prompt: The prompt to send to the model
            config: Generation configuration (types.GenerateContentConfig)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Response from the model
            
        Raises:
            Exception: If all retries fail
        '''
        last_exception: Exception | None = None
        
        for attempt in range(max_retries):
            try:
                await self._async_wait_for_rate_limit()
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    sleep_time = 2 ** attempt
                    await asyncio.sleep(sleep_time)
        
        raise last_exception  # type: ignore[misc]
    
    def create_summary_prompt(self, blog_text: str) -> str:
        '''
        Create an optimized prompt for Gemini.

        Args:
            blog_text: Blog post text
            
        Returns:
            Formatted prompt string
        '''
        return f'''You are a strict news editor. Output NO MORE THAN 2-3 very short sentences summarizing key points and events of the post.

If the text is one sentence long, just output the original text, DON'T ADD ANYTHING!

Summarize this blog post:
{blog_text}'''
    
    def summarize_blog_post(self, raw_blog_text: str) -> dict:
        '''
        Complete pipeline to summarize a Trump blog post.
        
        Args:
            raw_blog_text: Raw blog text (may contain tags)
            
        Returns:
            Dictionary with summary results and metadata
        '''
        start_time = time.time()
        
        if not raw_blog_text.strip():
            return {
                'success': False,
                'error': 'No content',
                'summary': '',
                'processing_time': 0.0,
            }
        
        try:
            text_length = len(raw_blog_text)
            word_count = len(raw_blog_text.split())
            
            if word_count <= 10:
                return {
                    'success': True,
                    'summary': raw_blog_text,
                    'warning': 'Text too brief to summarize, returning as is',
                    'word_count': word_count,
                    'processing_time': 0.0
                }
            
            # Generate summary using Gemini with retry logic
            prompt = self.create_summary_prompt(raw_blog_text)
            response = self._call_with_retry(prompt, self.generation_config)
            
            # Post-process the response
            summary = self.post_process_summary(response.text)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'summary': summary,
                'original_length': text_length,
                'word_count': word_count,
                'processing_time': round(processing_time, 2),
            }
            
        except Exception as e:
            return {
                'success': False,
                'summary': '',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def post_process_summary(self, raw_summary: str) -> str:
        '''
        Clean up the generated summary, removing model artifacts.
        
        Args:
            raw_summary: Raw output from the model
            
        Returns:
            Cleaned summary text
        '''
        summary = raw_summary.strip()
        
        if summary and summary.startswith('Trump:'):
            summary = re.sub('Trump: ', '', summary, flags=re.IGNORECASE)

        # Clean up whitespace
        summary = re.sub(r'\n\s*\n', '\n', summary)
        summary = summary.strip()
        
        # Ensure proper capitalization
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        return summary

    def create_title_prompt(self, blog_text: str) -> str:
        '''
        Create a prompt for generating a concise title.
        
        Args:
            blog_text: Blog post text
            
        Returns:
            Formatted prompt string for title generation
        '''
        return f'''You are a news headline writer. Create a single, short, concise headline that captures the main point.

Create a headline for this blog post:
{blog_text}'''
    
    def generate_title(self, raw_blog_text: str) -> dict:
        '''
        Generate a title for the blog post.
        
        Args:
            raw_blog_text: Blog post text
            
        Returns:
            Dictionary with title results and metadata
        '''
        start_time = time.time()
        
        word_count = len(raw_blog_text.split())
        if word_count <= 10:
            return {
                'success': True,
                'title': 'Empty blog post',
                'note': 'Text too brief to create a title',
                'word_count': word_count,
                'processing_time': 0.0
            }
        
        try:
            # Title-specific config - shorter output, slightly more creative
            title_config = types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.9,
                max_output_tokens=20,
                safety_settings=self.safety_settings,
            )
            
            prompt = self.create_title_prompt(raw_blog_text)
            response = self._call_with_retry(prompt, title_config)
            
            # Post-process the title
            title = self.post_process_title(response.text)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'title': title,
                'processing_time': round(processing_time, 2),
            }
            
        except Exception as e:
            return {
                'success': False,
                'title': '',
                'processing_time': time.time() - start_time,
                'error': str(e),
            }
    
    def post_process_title(self, raw_title: str) -> str:
        '''
        Clean up the generated title.
        
        Args:
            raw_title: Raw title output from the model
            
        Returns:
            Cleaned title text
        '''
        title = raw_title.strip()
        
        # Remove quotes if the model wrapped the title
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1].strip()
        
        # Ensure proper title case (capitalize first letter)
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]
        
        # Remove trailing punctuation that's not appropriate for titles
        while title and title[-1] in '.!*"':
            title = title[:-1].strip()
        
        while title and title[0] in '.!*"':
            title = title[1:].strip()
        
        return title.strip()

    def summarize_with_title(self, raw_blog_text: str) -> dict:
        '''
        Complete pipeline to generate both title and summary for a Trump blog post.
        
        Args:
            raw_blog_text: Raw blog text (may contain HTML tags)
            
        Returns:
            Dictionary with both title and summary results
        '''
        start_time = time.time()
        
        if not raw_blog_text.strip():
            return {
                'success': False,
                'title': '',
                'summary': '',
                'error': 'No content',
            }
        
        # Generate title
        title_result = self.generate_title(raw_blog_text)
        
        # Generate summary
        summary_result = self.summarize_blog_post(raw_blog_text)
        
        total_time = time.time() - start_time
        
        return {
            'success': title_result['success'] and summary_result['success'],
            'title': title_result.get('title', ''),
            'summary': summary_result.get('summary', ''),
            'title_processing_time': title_result.get('processing_time', 0),
            'summary_processing_time': summary_result.get('processing_time', 0),
            'total_processing_time': round(total_time, 2),
            'word_count': summary_result.get('word_count', 0),
            'errors': {
                'title_error': title_result.get('error'),
                'summary_error': summary_result.get('error')
            }
        }

    # ==================== ASYNC METHODS ====================
    
    async def summarize_blog_post_async(self, raw_blog_text: str) -> dict:
        '''
        Async version: Complete pipeline to summarize a Trump blog post.
        
        Args:
            raw_blog_text: Raw blog text (may contain tags)
            
        Returns:
            Dictionary with summary results and metadata
        '''
        start_time = time.time()
        
        if not raw_blog_text.strip():
            return {
                'success': False,
                'error': 'No content',
                'summary': '',
                'processing_time': 0.0,
            }
        
        try:
            text_length = len(raw_blog_text)
            word_count = len(raw_blog_text.split())
            
            if word_count <= 10:
                return {
                    'success': True,
                    'summary': raw_blog_text,
                    'warning': 'Text too brief to summarize, returning as is',
                    'word_count': word_count,
                    'processing_time': 0.0
                }
            
            # Generate summary using Gemini with async retry logic
            prompt = self.create_summary_prompt(raw_blog_text)
            response = await self._async_call_with_retry(prompt, self.generation_config)
            
            # Post-process the response
            summary = self.post_process_summary(response.text)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'summary': summary,
                'original_length': text_length,
                'word_count': word_count,
                'processing_time': round(processing_time, 2),
            }
            
        except Exception as e:
            return {
                'success': False,
                'summary': '',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def generate_title_async(self, raw_blog_text: str) -> dict:
        '''
        Async version: Generate a title for the blog post.
        
        Args:
            raw_blog_text: Blog post text
            
        Returns:
            Dictionary with title results and metadata
        '''
        start_time = time.time()
        
        word_count = len(raw_blog_text.split())
        if word_count <= 10:
            return {
                'success': True,
                'title': 'Empty blog post',
                'note': 'Text too brief to create a title',
                'word_count': word_count,
                'processing_time': 0.0
            }
        
        try:
            # Title-specific config - shorter output, slightly more creative
            title_config = types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.9,
                max_output_tokens=20,
                safety_settings=self.safety_settings,
            )
            
            prompt = self.create_title_prompt(raw_blog_text)
            response = await self._async_call_with_retry(prompt, title_config)
            
            # Post-process the title
            title = self.post_process_title(response.text)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'title': title,
                'processing_time': round(processing_time, 2),
            }
            
        except Exception as e:
            return {
                'success': False,
                'title': '',
                'processing_time': time.time() - start_time,
                'error': str(e),
            }
    
    async def summarize_with_title_async(self, raw_blog_text: str) -> dict:
        '''
        Async version: Complete pipeline to generate both title and summary.
        
        Args:
            raw_blog_text: Raw blog text (may contain HTML tags)
            
        Returns:
            Dictionary with both title and summary results
        '''
        start_time = time.time()
        
        if not raw_blog_text.strip():
            return {
                'success': False,
                'title': '',
                'summary': '',
                'error': 'No content',
            }
        
        # Generate title and summary concurrently
        title_result, summary_result = await asyncio.gather(
            self.generate_title_async(raw_blog_text),
            self.summarize_blog_post_async(raw_blog_text)
        )
        
        total_time = time.time() - start_time
        
        return {
            'success': title_result['success'] and summary_result['success'],
            'title': title_result.get('title', ''),
            'summary': summary_result.get('summary', ''),
            'title_processing_time': title_result.get('processing_time', 0),
            'summary_processing_time': summary_result.get('processing_time', 0),
            'total_processing_time': round(total_time, 2),
            'word_count': summary_result.get('word_count', 0),
            'errors': {
                'title_error': title_result.get('error'),
                'summary_error': summary_result.get('error')
            }
        }
    
    async def summarize_batch_async(self, posts: list[str]) -> list[dict]:
        '''
        Process multiple blog posts concurrently with rate limiting.
        
        Args:
            posts: List of blog post texts to summarize
            
        Returns:
            List of result dictionaries, one per post
        '''
        if not posts:
            return []
        
        # Process all posts concurrently - rate limiting is handled internally
        results = await asyncio.gather(
            *[self.summarize_with_title_async(post) for post in posts],
            return_exceptions=True
        )
        
        # Convert any exceptions to error results
        processed_results: list[dict] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'title': '',
                    'summary': '',
                    'error': str(result),
                    'post_index': i
                })
            elif isinstance(result, dict):
                result['post_index'] = i
                processed_results.append(result)
        
        return processed_results
