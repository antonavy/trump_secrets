# Trump Feed Financial Analyzer

This project monitors Donald Trump's social media feed (via `trumpstruth.org`), analyzes posts for financial market relevance using Google's Gemini Flash model, and sends real-time alerts to Telegram.

## Features

- **Real-time Monitoring**: Polls the RSS feed every 5 minutes.
- **AI Analysis**: Uses Gemini Flash to score posts on financial relevance (1-10), sentiment (Bullish/Bearish), and market impact.
- **Smart Notifications**:
  - 🚨 **MARKET MOVER**: High relevance (Score 8-10).
  - ⚠️ **MARKET RELEVANT**: Medium relevance (Score 4-7).
  - 📢 **Standard Post**: Low relevance (Score 1-3).
- **Structured Data**: Extracts tickers, sectors, and specific signals.

## Setup & Running

### Prerequisites

- Docker & Docker Compose
- A Google Gemini API Key
- A Telegram Bot Token & Chat ID

### Installation

1.  **Clone the repository**
2.  **Create a `.env` file** in the root directory with your credentials:
    ```bash
    GEMINI_API_KEY=your_gemini_key
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id
    TELEGRAM_CHANNEL_ID=your_channel_id  # Optional
    DB_FILENAME=trump_posts_db.json      # Optional
    ```
3.  **Prepare Data Directory**:
    Create a `data` folder and move your existing database (if any) into it:
    ```bash
    mkdir data
    mv trump_posts_db.json data/
    ```

### Running with Docker

Start the service in the background:

```bash
docker-compose up -d --build
```

View logs:

```bash
docker-compose logs -f
```

## Project Structure

- `trump_feed_parser.py`: Main service loop. Fetches feed and orchestrates analysis.
- `llm_summarize_v4_finance.py`: Contains the `TrumpFeedAnalyzer` class and Gemini integration logic.
- `feed_fetcher.py`: Helper for safe RSS fetching.
- `docker-compose.yml`: Container configuration.
