import ollama
import gc
import re
import time

class OptimizedQwen3Summarizer:
    def __init__(self):
        '''Initialize the Qwen3 summarizer with optimized settings.'''
        self.model = 'qwen3:1.7b'
        
        # Optimized settings for 4GB RAM machine
        self.base_options = {
            'num_ctx': 2048,       # Increased for longer blog posts from minimum 1536
            'num_predict': 80,     # Number of output tokens
            'temperature': 0.1,    # Less variety
            'top_p': 0.9,          # Focus on most likely tokens
            'repeat_penalty': 1.1, # Prevent repetition
            # 'stop': ['<|im_end|>', '<|im_start|>']  # ONLY ChatML tokens
        }
    
    def create_summary_prompt(self, blog_text: str) -> str:
        '''
        Create an optimized prompt for Qwen3 with proper ChatML formatting.

        Args:
            blog_text: Blog post text
            
        Returns:
            Formatted prompt string
        '''
        
        messages = [
            {
                'role': 'system',
                'content': 'You are a strict news editor. Output NO MORE THAN 2-3 very short sentences \
            summarizing key points and events of the post. \
            If the text is one sentence long, just output original text, DON\'T ADD ANYTHING!',
            },
            {
                'role': 'user',
                'content': f'Summarize this blog post:\n{blog_text}',
            },
            
        ]

        return messages
    
    def adjust_options_for_length(self, text_length: int) -> dict:
        '''
        Dynamically adjust generation options based on input text length.
        
        Args:
            text_length: Length of input text in characters
            
        Returns:
            Adjusted options dictionary
        '''
        options = self.base_options.copy()
        
        # Adjust context window based on text length
        if text_length < 600:
            options['num_ctx'] = 1536    # Minimum for short texts
        else:
            options['num_ctx'] = 2048    # Default for medium to long texts
        
        # Adjust output length based on input complexity
        if text_length < 600:
            options['num_predict'] = 80   # Minimum to avoid cut-offs
        else:
            options['num_predict'] = 96   # More length for medium to long texts
        
        return options
    
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
        
        # Step 1: Prepare optimized settings
        try:
            text_length = len(raw_blog_text)
        except Exception as e:
            return {
                'success': False,
                'summary': '',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        options = self.adjust_options_for_length(text_length)
        prompt = self.create_summary_prompt(raw_blog_text)
        
        word_count = len(raw_blog_text.split())
        if word_count <= 10:
            return {
                'success': True,
                'summary': raw_blog_text,  # Just return the original
                'warning': 'Text too brief to summarize, returning as is',
                'word_count': word_count,
                'processing_time': 0.0
            }
        else:
            try:
                # Step 2: Generate summary
                response = ollama.chat(
                    model=self.model,
                    messages=prompt,
                    options=options,
                    stream=False,
                    think=False,
                )
                
                # Step 3: Post-process the response
                summary = self.post_process_summary(response.message.content)
                
                processing_time = time.time() - start_time
                
                # Force garbage collection to free memory
                gc.collect()
                
                return {
                    'success': True,
                    'summary': summary,
                    'original_length': text_length,
                    'word_count': len(raw_blog_text.split()),
                    'processing_time': round(processing_time, 2),
                    'context_used': options['num_ctx'],
                    'tokens_generated': len(summary.split()) * 1.3  # Rough token estimate
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
        messages = [
            {
                'role': 'system',
                'content': 'You are a news headline writer. Create a single, short, concise\
                    headline that captures the main point.',
            },
            {
                'role': 'user',
                'content': f'Create a headline for this blog post:\n{blog_text}',
            },
            
        ]
        
        return messages
    
    def generate_title(self, raw_blog_text: str) -> dict:
        '''
        Generate a title for the blog post.
        
        Args:
            raw_blog_text: Blog post text
            
        Returns:
            Dictionary with title results and metadata
        '''
        start_time = time.time()
        
        # Title-specific options - shorter output, focused generation
        title_options = {
            'num_ctx': 1536,        # Smaller context for titles
            'num_predict': 16,      # ~8-12 words for a good headline
            'temperature': 0.2,     # Slightly more creative for engaging titles
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            # 'stop': ['<|im_end|>', '<|im_start|>']  # Stop at sentence end
        }
        
        prompt = self.create_title_prompt(raw_blog_text)

        word_count = len(raw_blog_text.split())
        if word_count <= 10:
            return {
                'success': True,
                'summary': 'Empty blog post',
                'note': 'Text too brief to create a title',
                'word_count': word_count,
                'processing_time': 0.0
            }
        else:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=prompt,
                    options=title_options,
                    stream=False,
                    think=False,
                )
                
                # Post-process the title
                title = self.post_process_title(response.message.content)
                
                processing_time = time.time() - start_time
                
                return {
                    'success': True,
                    'title': title,
                    'processing_time': round(processing_time, 2),
                    'tokens_used': len(title.split()) * 1.3  # Rough estimate
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
        
        # Step 1: Generate title
        title_result = self.generate_title(raw_blog_text)
        
        # Step 2: Generate summary
        summary_result = self.summarize_blog_post(raw_blog_text)
        
        total_time = time.time() - start_time
        
        # Force garbage collection
        gc.collect()
        
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
