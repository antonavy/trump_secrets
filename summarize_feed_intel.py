#!/usr/bin/env python3

import torch
import logging
import difflib  # For comparing text similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tinydb import TinyDB
import gc  # For garbage collection


# Configure logging
logging.basicConfig(
    filename='trump_feed_parser.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set device to CPU and optimize memory usage
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)  # Ensure float32 for CPU
torch.set_num_threads(2)  # Limit CPU threads to prevent memory issues
logger.info('Using CPU for inference with 2 threads')

# Constants
MODEL_ID = 'facebook/bart-large-cnn'  # Changed to a better model for summarization
MAX_LENGTH = 1024  # Increased to handle longer input texts
MIN_LENGTH = 50  # Increased minimum length for better summaries
MAX_SUMMARY_LENGTH = 200  # Increased for more comprehensive summaries
BATCH_SIZE = 1


def setup_model(model_id):
    '''Load model and tokenizer'''
    logger.info('Loading model and tokenizer...')
    try:
        # Clear any existing memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load model with memory optimizations
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            max_memory={0: "4GB"}  # Increased memory limit
        ).to(device)
        model.eval()  # Set to evaluation mode
        logger.info(f'Successfully loaded model: {model_id}')
        return tokenizer, model
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise


def format_summary(text):
    '''Format the summary text for better readability'''
    # Capitalize first letter of each sentence
    sentences = text.split('. ')
    formatted_sentences = [s.capitalize() for s in sentences if s.strip()]
    formatted_text = '. '.join(formatted_sentences)
    # Ensure proper ending punctuation
    if not formatted_text.endswith(('.', '!', '?')):
        formatted_text += '.'
    return formatted_text


def summarize_text(text, tokenizer, model):
    '''Summarize text using PyTorch model'''
    try:
        # Check if text is too short to summarize
        if len(text.strip()) < 80:
            logger.info(f'Text too short ({len(text)} chars), returning as is')
            return text.strip()

        # Log input text and its length
        logger.info(f'Input text length: {len(text)} characters')
        logger.debug(f'Input text: {text}')

        # Clear memory before processing
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Tokenize with proper padding
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True
        ).to(device)

        input_tokens = len(inputs["input_ids"][0])
        logger.info(f'Tokenized input length: {input_tokens} tokens')

        # Calculate max tokens for summary (approximately 1 token = 4 characters)
        max_tokens = MAX_SUMMARY_LENGTH // 3  # Increased ratio for better summaries
        min_tokens = MIN_LENGTH // 3  # Increased ratio for better summaries
        logger.info(f'Target max tokens for summary: {max_tokens}, min tokens: {min_tokens}')

        # Generate summary with optimized parameters
        try:
            with torch.no_grad():  # Disable gradient calculation
                # Try with advanced parameters first
                try:
                    logger.info("Attempting summarization with advanced parameters")
                    summary_ids = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_tokens,  # Use max_new_tokens instead of max_length
                        min_new_tokens=min_tokens,  # Ensure minimum length for meaningful summaries
                        num_beams=6,  # Increased for better search
                        length_penalty=2.0,  # Increased to encourage more complete summaries
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,  # Added to prevent repetition
                        temperature=0.7,  # Reduced for more focused generation
                        top_k=50,
                        top_p=0.9  # Slightly reduced for more focused generation
                    )
                except Exception as e:
                    # If advanced parameters fail, fall back to basic parameters
                    logger.warning(f"Advanced parameters failed: {e}. Falling back to basic parameters.")
                    summary_ids = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=max_tokens + input_tokens,  # Use max_length as fallback
                        min_length=min_tokens,
                        num_beams=4,
                        length_penalty=1.0,
                        early_stopping=True
                    )
        except Exception as e:
            logger.error(f"All summarization attempts failed: {e}")
            raise

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_tokens = len(summary_ids[0])
        logger.info(f'Generated summary tokens: {summary_tokens}')

        formatted_summary = format_summary(summary)
        logger.info(f'Final summary length: {len(formatted_summary)} characters')
        logger.info(f'Token to character ratio: {len(formatted_summary)/summary_tokens:.2f}')

        # Check if the summary is just the beginning of the text
        # More robust check: compare the first 50 characters and check similarity ratio
        import difflib
        is_beginning = False
        if len(formatted_summary) > 50:
            # Check if summary starts with beginning of text
            if text.lower().startswith(formatted_summary[:50].lower()):
                is_beginning = True
            # Check similarity ratio between summary and beginning of text
            similarity = difflib.SequenceMatcher(None, formatted_summary[:100].lower(), 
                                               text[:100].lower()).ratio()
            if similarity > 0.8:  # High similarity indicates summary is just beginning of text
                is_beginning = True

        if is_beginning:
            logger.warning("Summary appears to be just the beginning of the text. Regenerating with stricter parameters.")
            # Try again with more aggressive summarization parameters
            try:
                with torch.no_grad():
                    try:
                        logger.info("Attempting regeneration with stricter parameters")
                        summary_ids = model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=max_tokens,
                            min_new_tokens=min_tokens,
                            num_beams=8,  # Increased for better search
                            length_penalty=3.0,  # More aggressive length penalty
                            early_stopping=True,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.5,  # Increased to prevent repetition
                            temperature=0.6,  # Further reduced for more focused generation
                            top_p=0.85
                        )
                    except Exception as e:
                        # If stricter parameters fail, fall back to simpler parameters
                        logger.warning(f"Stricter parameters failed: {e}. Falling back to simpler parameters.")
                        summary_ids = model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=max_tokens + input_tokens,
                            min_length=min_tokens,
                            num_beams=4,
                            length_penalty=2.0,
                            early_stopping=True
                        )
            except Exception as e:
                logger.error(f"All regeneration attempts failed: {e}")
                # Continue with the original summary instead of failing
                logger.info("Using original summary as fallback")
                # No need to reassign summary_ids as it already contains the original summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted_summary = format_summary(summary)
            logger.info(f'Regenerated summary length: {len(formatted_summary)} characters')

        # Clear memory after processing
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Log summary and compression ratio
        logger.info(f'Compression ratio: {len(formatted_summary)/len(text):.2%}')
        logger.debug(f'Generated summary: {formatted_summary}')

        return formatted_summary

    except Exception as e:
        logger.error(f'Error during summarization: {e}')
        raise


def main(text: str) -> str:
    # Load model
    logger.info('Setting up model...')
    try:
        tokenizer, model = setup_model(MODEL_ID)
    except Exception as e:
        logger.error(f'Failed to set up model and tokenizer: {e}')
        return "Error: Failed to load summarization model"

    # Summarize
    logger.info('Summarizing...')
    try:
        summary = summarize_text(text, tokenizer, model)
        return summary
    except Exception as e:
        logger.error(f'Error during summarization: {e}')
        import traceback
        logger.error(traceback.format_exc())

        # Ultimate fallback: return a truncated version of the original text
        # if we can't generate a proper summary
        try:
            logger.warning("Summarization failed. Using text truncation as ultimate fallback.")
            # If text is short enough, return it as is
            if len(text) <= MAX_SUMMARY_LENGTH:
                return text

            # Otherwise, truncate to the first few sentences
            sentences = text.split('. ')
            truncated_text = ""
            for sentence in sentences:
                if len(truncated_text) + len(sentence) + 2 <= MAX_SUMMARY_LENGTH:
                    truncated_text += sentence + ". "
                else:
                    break

            if truncated_text:
                return truncated_text.strip()
            else:
                # If we couldn't get complete sentences, just truncate
                return text[:MAX_SUMMARY_LENGTH - 3] + "..."
        except:
            # If even the fallback fails, return the error message
            return "Error: Failed to generate summary"


if __name__ == '__main__':
    # Test case 1: Original test
    test_text1 = 'Today, I signed an Executive Order to launch the first-ever self-deportation program. Illegal aliens who stay in America face punishments, including—sudden deportation, in a place and manner solely of our discretion. TO ALL ILLEGAL ALIENS: BOOK YOUR FREE FLIGHT RIGHT NOW!'

    # Test case 2: Example from issue description
    test_text2 = "I am very proud of the strong and unwaveringly powerful leadership of India and Pakistan for having the strength, wisdom, and fortitude to fully know and understand that it was time to stop the current aggression that could have lead to to the death and destruction of so many, and so much. Millions of good and innocent people could have died! Your legacy is greatly enhanced by your brave actions. I am proud that the USA was able to help you arrive a this historic and heroic decision. While not even discussed, I am going to increase trade, substantially, with both of these great Nations. Additionally, I will work with you both to see if, after a \"thousand years,\" a solution can be arrived at concerning Kashmir. God Bless the leadership of India and Pakistan on a job well done!!!"

    # Run both tests
    for i, test_text in enumerate([test_text1, test_text2], 1):
        print(f"\nTest Case {i}:")
        print("\nOriginal text:")
        print("-" * 80)
        print(test_text)
        print("\nSummary:")
        print("-" * 80)
        print(main(test_text))
        print("-" * 80)
