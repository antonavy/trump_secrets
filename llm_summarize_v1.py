import requests
import json
import time
from typing import Optional

class TrumpBlogSummarizer:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the summarizer with Ollama connection.
        
        Args:
            ollama_url: URL where Ollama is running (default: localhost:11434)
        """
        self.ollama_url = ollama_url
        self.model = "qwen2.5:1.5b"
        
    def check_ollama_health(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def summarize_text(self, text: str, max_retries: int = 3) -> Optional[str]:
        """
        Summarize text using Ollama with Qwen model.
        
        Args:
            text: Text to summarize
            max_retries: Maximum number of retry attempts
            
        Returns:
            Summarized text or None if failed
        """
        if not self.check_ollama_health():
            print("Error: Ollama service is not running. Start it with 'ollama serve'")
            return None
        
        # Craft a focused prompt for consistent 2-3 sentence summaries
        prompt = f"""Summarize the following text in exactly 2-3 clear, concise sentences. Focus on the most important points and main message:

{text}

Summary:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Low temperature for consistent summaries
                "top_p": 0.9,
                "num_predict": 150,  # Limit output length
                "stop": ["\n\n", "Summary:", "Text:"]  # Stop tokens
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=60  # Increased timeout for processing
                )
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result.get('response', '').strip()
                    
                    # Clean up the summary
                    summary = self._clean_summary(summary)
                    
                    if summary:
                        return summary
                    else:
                        print(f"Attempt {attempt + 1}: Empty response received")
                else:
                    print(f"Attempt {attempt + 1}: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1}: Request failed - {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
        
        return None
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the summary output."""
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Summary:", "In summary,", "To summarize,", 
            "The text discusses", "This text", "The article"
        ]
        
        summary = summary.strip()
        
        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()
        
        # Ensure proper sentence structure
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        return summary
    
    def process_trump_blog_post(self, blog_text: str) -> dict:
        """
        Process a Trump blog post and return structured result.
        
        Args:
            blog_text: Raw blog post text
            
        Returns:
            Dictionary with original text length, summary, and processing info
        """
        if not blog_text.strip():
            return {"error": "Empty text provided"}
        
        # Basic preprocessing - remove excessive whitespace
        cleaned_text = ' '.join(blog_text.split())
        
        # Skip summarization for very short texts
        if len(cleaned_text.split()) < 20:
            return {
                "original_length": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "summary": cleaned_text,
                "note": "Text too short to summarize"
            }
        
        print(f"Processing text ({len(cleaned_text.split())} words)...")
        start_time = time.time()
        
        summary = self.summarize_text(cleaned_text)
        
        processing_time = time.time() - start_time
        
        return {
            "original_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "summary": summary,
            "processing_time": round(processing_time, 2),
            "success": summary is not None
        }

# Example usage
if __name__ == "__main__":
    # Initialize the summarizer
    summarizer = TrumpBlogSummarizer()
    
    # Example Trump blog text (replace with actual content)
    sample_text = """
    The Fake News Media continues to write phony stories about me and my campaign. 
    They don't want to report the truth about how well we're doing in the polls. 
    Our rallies are packed with thousands of patriots who love America. 
    The corrupt establishment is trying everything to stop us, but we will not be stopped. 
    We will Make America Great Again and bring back jobs, secure our borders, and restore law and order. 
    The American people are tired of the lies and deception from the mainstream media.
    """
    
    # Process the text
    result = summarizer.process_trump_blog_post(sample_text)
    
    if result.get("success"):
        print("\n" + "="*50)
        print("SUMMARY RESULT")
        print("="*50)
        print(f"Original text: {result['word_count']} words")
        print(f"Processing time: {result['processing_time']}s")
        print(f"\nSummary:\n{result['summary']}")
    else:
        print(f"Summarization failed: {result}")