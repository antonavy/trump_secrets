from google import genai
from google.genai import types
import asyncio
import os
import json
import time
from collections import deque
from pydantic import BaseModel, Field
from typing import List, Optional

# --- DATA STRUCTURES FOR FINANCIAL ANALYSIS ---
class MarketImpact(BaseModel):
    tickers: List[str] = Field(description="Crypto tickers explicitly mentioned or implied (e.g., 'BTC', 'ETH', 'XRP')")
    sectors: List[str] = Field(description="Sectors affected (e.g., 'Automotive', 'Defense', 'Crypto')")
    signal: str = Field(description="Explanation of effect on markets (e.g., 'positive for tech stocks due to deregulation')")
    sentiment: str = Field(description="Sentiment regarding the economy/market: BULLISH, BEARISH, or NEUTRAL")
    relevance_score: int = Field(description="Relevance to finance and crypto on a scale of 1 to 10")

class PostAnalysis(BaseModel):
    post_id: int = Field(description="The index of the post in the provided list")
    headline: str = Field(description="A really short headline")
    summary: str = Field(description="Concise summary covering the event and news context")
    is_financial_relevant: bool = Field(description="True if the post is relevant to financial markets")
    market_impact: Optional[MarketImpact] = Field(description="Detailed market analysis if relevant")

class AnalysisResult(BaseModel):
    analyses: List[PostAnalysis]

# --- MAIN CLASS ---
class TrumpFeedAnalyzer:
    def __init__(self, api_key: str | None = None, model_name: str = 'gemini-flash-latest', max_rpm: int = 15):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError('GEMINI_API_KEY needed')
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        # Rate limiting
        self.max_rpm = max_rpm
        self._request_timestamps = deque()
        self._rate_limit_lock = asyncio.Lock()

        # Relaxed safety settings are required for political content
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

    async def _wait_for_rate_limit(self):
        async with self._rate_limit_lock:
            current_time = time.time()
            while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
                self._request_timestamps.popleft()
            
            if len(self._request_timestamps) >= self.max_rpm:
                sleep_time = 60 - (current_time - self._request_timestamps[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                current_time = time.time()
                while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
                    self._request_timestamps.popleft()
            
            self._request_timestamps.append(time.time())

    async def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig, max_retries: int = 3):
        last_exception: Exception | None = None
        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    await asyncio.sleep(sleep_time)
        raise last_exception  # type: ignore[misc]

    async def analyze_batch(self, posts: List[str], batch_size: int = 20) -> List[dict]:
        """
        Sends a batch of posts to Gemini, chunking them to avoid context limits.
        Returns structured JSON data ready for your financial DB.
        """
        if not posts:
            return []

        all_results = []
        
        # Process in chunks
        for i in range(0, len(posts), batch_size):
            chunk = posts[i:i + batch_size]
            chunk_start_index = i
            
            # Prepare input with GLOBAL IDs to map back correctly
            formatted_posts = "\n".join([f"POST_ID {chunk_start_index + j}: {p}" for j, p in enumerate(chunk)])

            prompt = f"""
            You are a high-frequency trading algorithm's sentiment analysis engine. 
            Your job is to analyze social media posts from Donald Trump for market-moving information.

            RELEVANCE SCORING CRITERIA (1-10):
            10: Direct policy announcement affecting specific companies/sectors (tariffs, regulations, executive orders)
            8-9: Threats or promises of future policy with named targets (companies, countries, industries)
            6-7: General economic/market commentary with implications (Fed policy, inflation, trade deals)
            4-5: Indirect business impact (appointments of regulators, geopolitical tensions)
            2-3: Campaign rhetoric mentioning economy/jobs without specifics
            1: Pure politics, personal attacks, or slogans with zero market relevance

            SENTIMENT CLASSIFICATION:
            BULLISH: Tax cuts, deregulation, pro-business appointments, trade deal progress, "great economy" statements
            BEARISH: Tariff threats, sanctions, regulatory crackdowns, company-specific criticism, recession warnings
            NEUTRAL: Mixed signals, non-committal statements, or irrelevant content

            RISK LEVEL (1-10):
            10: Immediate executive action announced (tariffs effective now, emergency powers)
            8-9: Credible threat with timeline (will impose X by Y date)
            6-7: Policy direction signaled without timeline
            4-5: Criticism of company/sector without action
            2-3: Vague economic commentary
            1: No market impact

            For each post:
            1. Assign relevance_score using the scale above.
            2. Set is_financial_relevant=true ONLY if relevance_score >= 4.
            3. If the post is financially relevant, explain the effect on markets in a concise 'signal' field.
            4. Generate a punchy 'headline' (3-5 words max).
            5. Write a strictly factual 'summary' of the content.
               - CRITICAL: Do NOT include meta-commentary about market relevance (e.g., "This has no market impact").
               - CRITICAL: Do NOT explain the sentiment or signal in the summary.
               - Just summarize what was said or done.
            6. If relevant: extract tickers (use "SECTOR" tags like TECH, AUTO if no specific ticker), sentiment (BULLISH/BEARISH/NEUTRAL), and risk_level.
            POSTS:
            {formatted_posts}
            """

            try:
                response = await self._call_with_retry(
                    prompt=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        safety_settings=self.safety_settings,
                        response_mime_type="application/json",
                        response_schema=AnalysisResult, 
                    )
                )
                
                if response.text:
                    result_data = json.loads(response.text)
                    chunk_analyses = result_data.get('analyses', [])
                    all_results.extend(chunk_analyses)
                else:
                    print(f"Warning: Empty response for chunk {i}-{i+batch_size}")

            except Exception as e:
                print(f"Error processing chunk {i}-{i+batch_size}: {e}")
                continue
                
        return all_results
