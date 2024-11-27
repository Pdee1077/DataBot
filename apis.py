import os
import requests
from dotenv import load_dotenv
import logging
from openai import OpenAIError
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import openai
import sqlite3

# Load environment variables
load_dotenv("C:/bot/.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")


# Define your API functions here
def connect_to_database():
    db_path = r"C:\sqlite-tools\datanew.db"  # Replace with your database path
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.info("Successfully connected to the database.")
        return conn, cursor
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None, None

def get_perplexity_response(query):
    # Implementation here
    pass


def fetch_tweets_twitter_api(hashtag):
    # Implementation here
    pass

def handle_langchain_query(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Answer the following question: {query}",
    )
    llm = OpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.run(query)
        return response
    except Exception as e:
        return f"Error with LangChain: {e}"

def get_chatgpt_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.OpenAIError as e:
        return f"Error with OpenAI API: {e}"


def get_openai_response(prompt):
    # Implementation here
    pass


# Function to interact with OpenAI
def get_openai_response(prompt):
    """Fetch a response from the OpenAI API using the o1-preview or o1-mini model."""
    try:
        openai.api_key = OPENAI_API_KEY  # Use the key loaded from .env
        response = openai.ChatCompletion.create(
            model="o1-preview",  # Switch between 'o1-preview' or 'o1-mini' as needed
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        try:
            return response["choices"][0]["message"]["content"]
        except KeyError as e:
            logger.error(f"Unexpected response structure from OpenAI API: {e}")
            return "Error: Unexpected response from API"
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error when calling OpenAI API: {e}")
            return "An unexpected error occurred"
    except Exception as e:
        logger.error(f"Error creating OpenAI chat completion: {e}")
        return "An error occurred while processing your request"


# Function to interact with Perplexity API
def get_perplexity_response(prompt):
    """Fetch a response from the Perplexity API."""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "question": prompt,  # Adjust data format according to the actual Perplexity API requirements
            "max_tokens": 50,  # Optional: Adjust max_tokens as needed
        }
        response = requests.post(
            "https://api.perplexity.ai/v1/answer", headers=headers, json=data
        )
        response.raise_for_status()
        return response.json().get("answer", "No response available.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Perplexity API error: {e}")
        return f"Perplexity API error: {e}"


def fetch_tweets_twitter_api(hashtag):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    params = {
        "query": f"#{hashtag}",
        "max_results": 10,
        "tweet.fields": "created_at,text,author_id,source",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        tweets = []
        for tweet in data.get("data", []):
            created_at = tweet.get("created_at", "Unknown time")
            text = tweet.get("text", "No content available")
            author_id = tweet.get("author_id", "Unknown user")
            source = tweet.get("source", "Unknown source")

            formatted_tweet = (
                f"<strong>User ID:</strong> {author_id}<br>"
                f"<strong>Tweet:</strong> {text}<br>"
                f"<strong>Date/Time:</strong> {created_at}<br>"
                f"<strong>Source:</strong> {source}<br>"
            )
            tweets.append(formatted_tweet)

        return tweets if tweets else ["No results found."]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching tweets: {e}")
        return [f"Error fetching tweets: {e}"]
    except KeyError as e:
        logger.error(f"Unexpected response format: {e}")
        return ["Error: Unexpected response format from Twitter API."]

def get_chatgpt_response(query, model="gpt-4", temperature=0.7, max_tokens=200):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"].strip()
    except OpenAIError as e:
        logger.error(f"ChatGPT API error: {e}")
        return f"Error: Unable to fetch response from ChatGPT."

def analyze_database_with_langchain(query, params):
    """
    Fetches database results and analyzes them using LangChain.
    """
    # Step 1: Fetch database results
    results = fetch_database_results(query, params)
    if isinstance(results, str):
        return results  # Return error message if fetching fails

    # Step 2: Format results for LangChain
    formatted_data = "\n".join(", ".join(str(item) for item in row) for row in results)
    prompt = f"Analyze the following database results:\n{formatted_data}\nProvide insights."

    # Step 3: Use LangChain to analyze data
    return handle_langchain_query(prompt)

def fetch_database_results(query, params):
    db_path = r"C:\bot\chatbot_app\your_database.db"  # Ensure this is correct
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results if results else "No data found for the given query."
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return f"Database error: {e}"


if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set. Check your .env file.")
if not PERPLEXITY_API_KEY:
    logger.error("PERPLEXITY_API_KEY not set. Check your .env file.")
if not TWITTER_BEARER_TOKEN:
    logger.error("TWITTER_BEARER_TOKEN not set. Check your .env file.")

