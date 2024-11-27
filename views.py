from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from django import forms
from django.db import models
import json
import logging
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import requests
import openai
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from django.http import HttpResponse
from django.http import JsonResponse
from .apis import connect_to_database
from django.conf import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
dotenv_path = r"C:\bot\.env" 


# Environment Variables
DATABASE_PATH = os.getenv("DATABASE_PATH", r"C:\Sqlite-tools\datanew.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
NASDAQ_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GOOGLE_API_KEY: os.getenv("GOOGLE_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER_TOKEN:
    raise ValueError("Twitter Bearer Token is missing.")

import requests

def fetch_tweets(query):
    url = "https://real-time-x-com-data-scraper.p.rapidapi.com/Search/"
    params = {"q": query, "count": "20", "language": "en"}
    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return [tweet.get('content', 'No content') for tweet in data.get('posts', [])]
    except requests.exceptions.RequestException as e:
        return [f"Error fetching tweets: {e}"]


def validate_config():
    missing_keys = []
    for key_name, key_value in {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "NASDAQ_API_KEY": os.getenv("NASDAQ_API_KEY"),
        "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),
        "RAPIDAPI_KEY": os.getenv("RAPIDAPI_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "COOKING_API_KEY": os.getenv("COOKING_API_KEY"),
    }.items():
        if not key_value:
            missing_keys.append(key_name)
    if missing_keys:
        logger.error(f"Missing API keys: {', '.join(missing_keys)}")
    else:
        logger.info("All API keys are set correctly.")

# Call the function at the appropriate place in your code
validate_config()

def test_env_variables(request):
    env_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "NASDAQ_API_KEY": os.getenv("NASDAQ_API_KEY"),
        "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),
        "RAPIDAPI_KEY": os.getenv("RAPIDAPI_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "COOKING_API_KEY": os.getenv("COOKING_API_KEY"),
    }
    return JsonResponse(env_keys)


# settings.py or views.py
import os
from dotenv import load_dotenv

load_dotenv()  # Ensure the .env file is loaded
NASDAQ_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")

def fetch_nasdaq_data(symbol):
    NASDAQ_API_URL = "https://data.nasdaq.com/api/v3/datasets/WIKI/{symbol}.json"
    NASDAQ_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")

    if not NASDAQ_API_KEY:
        return {"error": "API key is missing. Please set NASDAQ_DATA_LINK_API_KEY in your .env file."}

    url = NASDAQ_API_URL.format(symbol=symbol)
    params = {"api_key": NASDAQ_API_KEY}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

def handle_nasdaq_command(symbol):
    """
    Handle the /nasdaq command to fetch stock data.
    """
    if not symbol:
        return "Please provide a stock symbol after /nasdaq."

    data = fetch_nasdaq_data(symbol)
    if "error" in data:
        return data["error"]

    # Format the response
    dataset = data.get("dataset", {})
    name = dataset.get("name", "Unknown Dataset")
    description = dataset.get("description", "No description available")
    latest_data = dataset.get("data", [])[0] if dataset.get("data") else "No data available"

    response = (
        f"Stock Data for {symbol}:\n"
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"Latest Data: {latest_data}"
    )
    return response

COMMAND_HANDLERS = {

    "/nasdaq": handle_nasdaq_command,  # Add Nasdaq handler
}



def error_404(request, exception):
    """Custom 404 error handler."""
    return render(request, '404.html', status=404)

def error_500(request):
    """Custom 500 error handler."""
    return render(request, '500.html', status=500)

def get_realtime_data(request, symbol):
    # Simulating real-time data for 'AAPL'
    if symbol.upper() == "AAPL":
        data = {
            "symbol": "AAPL",
            "price": 150.75,
            "volume": 5000000
        }
        return JsonResponse(data)
    else:
        return JsonResponse({"error": "No data available for this symbol"}, status=404)

def format_data_as_table(data, headers):
    """
    Formats a list of tuples into a tabular string.
    Args:
        data: List of tuples (rows of data).
        headers: List of column headers.
    Returns:
        A formatted string representing the data in a table.
    """
    if not data:
        return "No data available to display."
    
    # Use tabulate to create a table-like format
    return tabulate(data, headers=headers, tablefmt="grid")  # Use 'plain' for simpler formatting

def handle_google_command(query):
    """Handles requests to the Google API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: Google API Key is missing. Please set the GOOGLE_API_KEY environment variable."

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    payload = {
        "contents": [
            {"parts": [{"text": query}]}
        ]
    }
    headers = {
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(f"{url}?key={api_key}", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates")
        if candidates and candidates[0].get("content"):
            parts = candidates[0]["content"].get("parts")
            if parts and parts[0].get("text"):
                return parts[0]["text"]
            else:
                return "Error: Unexpected response format (missing text in parts)."
        else:
            return f"Error: Unexpected response format. {data}"

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"



def main_command_handler(command):
    command_type, *args = command.split(":", 1)
    handler = COMMAND_HANDLERS.get(command_type.strip().lower())
    if handler:
        return handler(args[0].strip() if args else "")
    else:
        return "Command not recognized. Use /help for a list of commands."

def compare_stocks(command):
    try:
        # Extract stock symbols from the command
        _, stocks = command.split(" ", 1)
        stock_symbols = stocks.split()

        if len(stock_symbols) != 2:
            return {"error": "Please provide exactly two stock symbols to compare."}

        symbol1, symbol2 = stock_symbols

        # Fetch stock data from the database or an API
        data1 = fetch_stock_data(symbol1)  # Replace with your data fetching logic
        data2 = fetch_stock_data(symbol2)  # Replace with your data fetching logic

        # Compare data and format the response
        comparison = {
            "Symbol 1": symbol1,
            "Data 1": data1,
            "Symbol 2": symbol2,
            "Data 2": data2,
        }
        return {"reply": comparison, "api_used": "Database/API"}
    except Exception as e:
        return {"error": str(e)}

def fetch_stock_data(symbol):
    # Placeholder for fetching data from database or API
    # Example query from the database:
    with connect_to_database() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stock_data WHERE symbol = ? LIMIT 1", (symbol,))
        return cursor.fetchone() or {"error": f"No data found for {symbol}"}


def handle_slash_command(user_input):
    logger.info(f"User input: {user_input}")
    if user_input.startswith("/google:"):
        query = user_input.split(":", 1)[1].strip()
        response_text = handle_google_command(query)
        return {"reply": response_text, "api_used": "Google API"}



    # Other commands
    elif user_input == "/test-db":
        # Example for testing the database
        return {"reply": "Database connection successful.", "api_used": "Database"}
    elif user_input.startswith("/run-chain-db:"):
        symbol = user_input.split(":", 1)[1].strip()  # Extract the symbol
        # Call the LangChain function here
        response_text = handle_langchain_command(symbol)
        return {"reply": response_text, "api_used": "LangChain"}
    elif user_input.startswith("/trend-range"):
        # Handle trend-range logic here
        return {"reply": "Trend-range command executed.", "api_used": "Trend Analysis"}
    
    else:
        return {
            "reply": "Command not recognized. Available commands: /test-db, /google:[query], etc.",
            "api_used": "N/A",
        }

import os
import requests

def handle_google_command(query):
    """Handles requests to the Google API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: Google API key not configured. Please check your .env file."

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": query}]}
        ]
    }

    try:
        response = requests.post(f"{url}?key={api_key}", json=payload, headers=headers)
        response.raise_for_status()  # Raise HTTP errors
        data = response.json()
        # Extract the response text
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response content found.")
        )
    except requests.RequestException as e:
        return f"Error with Google API: {str(e)}"
def google_query(request, query):
    """Handles requests to the Google API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return JsonResponse({"error": "Google API Key is missing. Please check your .env file."}, status=400)

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": query}]}]
    }

    try:
        response = requests.post(f"{url}?key={api_key}", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response content found.")
        )
        return JsonResponse({"reply": content})
    except requests.RequestException as e:
        return JsonResponse({"error": f"Error: {e}"}, status=500)

def handle_slash_command(user_input):
    if user_input.startswith("/tweets:"):
        hashtag = user_input.split(":", 1)[1].strip()
        return {"reply": fetch_tweets(hashtag), "api_used": "X.com API"}
    else:
        return {"reply": "Command not recognized. Use /help for available commands.", "api_used": "N/A"}


def handle_slash_command(user_input):
    if user_input.startswith("/google:"):
        query = user_input.split(":", 1)[1].strip()
        response_text = handle_google_command(query)
        return {"reply": response_text, "api_used": "Google API"}
    # ... handle other commands ...


    elif user_input == "/test-db":
        return {"reply": "Database connection successful.", "api_used": "Database"}
    
    elif user_input.startswith("/run-chain-db:"):
        symbol = user_input.split(":", 1)[1].strip()
        response_text = handle_langchain_command(symbol)
        return {"reply": response_text, "api_used": "LangChain"}
    
    elif user_input.startswith("/nasdaq:"):
        # Handle Nasdaq stock data
        symbol = user_input.split(":")[1].strip().upper()
        data = fetch_nasdaq_data(symbol)
        if "error" in data:
            return {"reply": data["error"], "api_used": "Nasdaq API"}
        
        stock_data = data.get("dataset", {})
        name = stock_data.get("name", "N/A")
        description = stock_data.get("description", "N/A")
        latest_data = stock_data.get("data", [])[0]  # Assuming the first entry is the latest
        response = f"Stock: {name}\nDescription: {description}\nLatest Data: {latest_data}"
        return {"reply": response, "api_used": "Nasdaq API"}

    else:
        return {"reply": "Command not recognized. Available commands: /test-db, /google:[query], etc.", "api_used": "N/A"}


# Fetch data from Nasdaq API for a given stock symbol
def fetch_nasdaq_data(symbol):
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        return {"error": "API key not found. Please set NASDAQ_DATA_LINK_API_KEY in the .env file."}

    url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/{symbol}.json?api_key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data  # Return the JSON response for further processing
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


# Chatbot view that processes commands
def chatbot_response(request):
    user_input = request.POST.get('message', '').strip()  # Get user input from the POST request

    # Initialize responses
    response_data = {"reply": "Error: No response generated.", "api_used": "N/A"}
    
    if user_input:
        # Call the handle_slash_command function to process the command
        response_data = handle_slash_command(user_input)
    else:
        response_data = {"reply": "No command provided", "api_used": "N/A"}

    return JsonResponse(response_data)  # Return the response as JSON

def handle_slash_command(command):
    if command.startswith("/nasdaq:"):
        # Extract the stock symbol from the command
        symbol = command.split(":")[1].strip().upper()
        if not symbol:
            return {"error": "Please provide a stock symbol after /nasdaq:."}

        # Fetch data using the helper function
        data = fetch_nasdaq_data(symbol)
        if "error" in data:
            return {"reply": data["error"], "api_used": "Nasdaq API"}

        # Process and format the response
        stock_data = data.get("dataset", {})
        name = stock_data.get("name", "N/A")
        description = stock_data.get("description", "N/A")
        latest_data = stock_data.get("data", [])[0]  # Assuming the first entry is the latest

        response = (
            f"Stock: {name}\n"
            f"Description: {description}\n"
            f"Latest Data: {latest_data}"
        )
        return {"reply": response, "api_used": "Nasdaq API"}
    else:
        return {"error": "Unknown slash command."}

def handle_nasdaq_command(symbol):
    if not symbol:
        return {"reply": "Please provide a stock symbol after /nasdaq:.", "api_used": "Nasdaq API"}

    # Fetch data using the helper function
    data = fetch_nasdaq_data(symbol)
    if "error" in data:
        return {"reply": data["error"], "api_used": "Nasdaq API"}

    # Process and format the response
    stock_data = data.get("dataset", {})
    name = stock_data.get("name", "N/A")
    description = stock_data.get("description", "N/A")
    latest_data = stock_data.get("data", [])[0] if stock_data.get("data") else "No data available"

    response = (
        f"Stock: {name}\n"
        f"Description: {description}\n"
        f"Latest Data: {latest_data}"
    )
    return {"reply": response, "api_used": "Nasdaq API"}


def fetch_nasdaq_data(symbol):
    """
    Fetch data from Nasdaq API for a given stock symbol.
    """
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        return {"error": "API key not found. Please set NASDAQ_DATA_LINK_API_KEY in the .env file."}

    url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/{symbol}.json?api_key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data  # Return the JSON response for further processing
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def connect_to_database():
    db_path = r"C:\sqlite-tools\datanew.db"  # Ensure this path matches your actual database path
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.info("Successfully connected to the database.")
        return conn, cursor
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None, None


@csrf_exempt
def chatbot_response(request):
    user_input = request.POST.get("message", "").strip()

    if not user_input:
        return JsonResponse({"reply": "No input received. Please type a message."})

    # Check for slash commands
    if user_input.startswith("/"):
        response = handle_slash_command(user_input)
        return JsonResponse(response)  # Ensure the response is returned correctly

    # Default behavior for unrecognized commands
    return JsonResponse({"reply": "Command not recognized. Use /help for available commands.", "api_used": "N/A"})


    # Check if the command exists in the handlers
    handler = COMMAND_HANDLERS.get(command)
    if handler:
        try:
            response = handler(argument)  # Call the command's handler with the argument
            return JsonResponse({"reply": response, "api_used": command})
        except Exception as e:
            return JsonResponse({"reply": f"Error processing {command}: {str(e)}", "api_used": command})
    else:
        # If the command is not recognized
        return JsonResponse({
            "reply": "Command not recognized. Use /help for available commands.",
            "api_used": "N/A"
        })

def chatbot_response(request):
    user_input = request.POST.get("message", "").strip()

    if not user_input:
        return JsonResponse({"reply": "No input received. Please type a message."})

    # Define responses to collect
    openai_response = {"reply": "Error: OpenAI response not available.", "api_used": "OpenAI"}
    perplexity_response = {"reply": "Error: Perplexity response not available.", "api_used": "Perplexity"}
    langchain_response = {"reply": "Error: LangChain response not available.", "api_used": "LangChain"}

    # Threads to execute API calls concurrently
    def query_openai():
        nonlocal openai_response
        openai_response["reply"] = get_openai_response(user_input)

    def query_perplexity():
        nonlocal perplexity_response
        perplexity_response["reply"] = get_perplexity_response(user_input)

    def query_langchain():
        nonlocal langchain_response
        langchain_response["reply"] = run_langchain_query(user_input)

    # Start threads
    threads = [
        threading.Thread(target=query_openai),
        threading.Thread(target=query_perplexity),
        threading.Thread(target=query_langchain)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Combine responses into one message
    combined_reply = (
        f"**OpenAI**: {openai_response['reply']}\n\n"
        f"**Perplexity**: {perplexity_response['reply']}\n\n"
        f"**LangChain**: {langchain_response['reply']}"
    )

    return JsonResponse({"reply": combined_reply})



def get_openai_response(question):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=question,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return "OpenAI could not process your request."

def get_perplexity_response(question):
    try:
        # Replace `PERPLEXITY_API_URL` and `PERPLEXITY_API_KEY` with your actual values
        url = "https://api.perplexity.ai/answer"
        headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}
        payload = {"question": question}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "Perplexity did not return an answer.")
    except Exception as e:
        logger.error(f"Perplexity API Error: {e}")
        return "Perplexity could not process your request."

def run_langchain_query(question):
    try:
        # Example using LangChain with OpenAI model
        llm = OpenAI(model="text-davinci-003", api_key=OPENAI_API_KEY)
        chain = LLMChain(
            prompt=PromptTemplate(
                input_variables=["query"],
                template="Answer the following question: {query}"
            ),
            llm=llm
        )
        return chain.run(question)
    except Exception as e:
        logger.error(f"LangChain Error: {e}")
        return "LangChain could not process your request."





def my_view(request):
    return HttpResponse("This is a placeholder for my_view. Replace it with actual functionality.")

# Building model
class Building(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)

    def __str__(self):
        return self.name

# Building form
class BuildingForm(forms.ModelForm):
    class Meta:
        model = Building
        fields = ["name", "location"]

# Building creation view
class BuildingCreateView(CreateView):
    model = Building
    form_class = BuildingForm
    template_name = "add_building.html"  # Ensure this template exists
    success_url = reverse_lazy("chat_view")





# Validate environment variables
if not os.path.exists(DATABASE_PATH):
    logger.error(f"Database file not found at {DATABASE_PATH}. Please check the path.")
else:
    logger.info(f"Database file found at {DATABASE_PATH}")

if not OPENAI_API_KEY:
    logger.error("Error: OPENAI_API_KEY not set in .env")
if not PERPLEXITY_API_KEY:
    logger.error("Error: PERPLEXITY_API_KEY not set in .env")
if not TWITTER_BEARER_TOKEN:
    logger.error("Error: TWITTER_BEARER_TOKEN not set in .env")

# Import custom modules from apis.py
from .apis import (
    get_openai_response,
    get_perplexity_response,
    fetch_tweets_twitter_api,
    fetch_database_results,
)

# SQLite database path
db_path = r"C:\Sqlite-tools\datanew.db"

try:
    # Ensure the database path is valid
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print("Connected to the database successfully.")

    # Test query
    test_symbol = "AAPL"  # Change this to test a different symbol
    cursor.execute(
        """
        SELECT date, symbol, close_price, ema_12, ema_26
        FROM stock_data
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 25
        """,
        (test_symbol,),
    )
    results = cursor.fetchall()

    # Check if results exist
    if results:
        print(f"Latest data for symbol {test_symbol}:")
        for row in results:
            print(row)
    else:
        print(f"No data found for symbol: {test_symbol}")

except FileNotFoundError as fnfe:
    print(fnfe)

except sqlite3.Error as e:
    print(f"Database error: {e}")

finally:
    # Ensure the connection is closed
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connection closed.")

def compare_stocks(command):
    try:
        # Extract stock symbols
        _, stocks = command.split(" ", 1)
        stock_symbols = stocks.split()

        if len(stock_symbols) != 2:
            return {"reply": "Please provide exactly two stock symbols to compare.", "api_used": "N/A"}

        symbol1, symbol2 = stock_symbols

        # Fetch data for both stocks
        data1 = fetch_stock_data(symbol1)
        data2 = fetch_stock_data(symbol2)

        if "error" in data1 or "error" in data2:
            return {
                "reply": f"Error fetching data: {data1.get('error', '')} {data2.get('error', '')}",
                "api_used": "N/A"
            }

        # Format comparison output
        reply = (
            f"Comparison of {symbol1} and {symbol2}:\n\n"
            f"{symbol1}: {data1}\n"
            f"{symbol2}: {data2}"
        )

        return {"reply": reply, "api_used": "Database/API"}
    except Exception as e:
        return {"reply": f"Error processing command: {str(e)}", "api_used": "N/A"}

def fetch_stock_data(symbol):
    # Placeholder for fetching data
    with connect_to_database() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stock_data WHERE symbol = ? LIMIT 1", (symbol,))
        row = cursor.fetchone()
        if row:
            return dict(zip([desc[0] for desc in cursor.description], row))
        return {"error": f"No data found for symbol {symbol}"}


def handle_slash_command(command):
    if command.startswith("/compare"):
        return compare_stocks(command)
    # Handle /AAPL command (or other stock-related command)
    elif command.startswith("/AAPL"):
        return process_stock_command(command)
    elif command.startswith("/test-db"):
        return test_database_connection()
    elif command.startswith("/run-chain-db"):
        return run_chain_analysis(command)
    elif command.startswith("/trend-range"):
        return trend_range_analysis(command)
    elif command.startswith("/sector"):
        return sector_analysis(command)
    elif command.startswith("/emaB0-1"):
        return ema_analysis(command)
    else:
        return {"reply": "Command not recognized. Available commands:\n- `/test-db`: Test database connection.\n- `/run-chain-db:[symbol]`: Use LangChain to analyze stock data.\n- `/trend-range MIN MAX`: Stocks with trends in the range.\n- `/sector:[sector]`: List stocks in a sector.\n- `/emaB0-1`: Stocks with EMA difference between 0% and 1%.\n- `Ask:[question]`: Ask OpenAI a question.\n", "api_used": "N/A"}




def call_google_api(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    api_key = os.getenv("GOOGLE_API_KEY")

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(f"{url}?key={api_key}", json=payload, headers=headers)
        response.raise_for_status()  # Raise error for HTTP codes 4xx/5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def handle_openai_request(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}]
        )
        return {"reply": response['choices'][0]['message']['content'], "api_used": "OpenAI"}
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {"reply": "An error occurred while processing your question.", "api_used": "OpenAI"}



          
import threading

def ask_all_apis(question):
    # Responses from APIs
    openai_reply = "Error: Unable to query OpenAI."
    perplexity_reply = "Error: Unable to query Perplexity."
    langchain_reply = "Error: Unable to query LangChain."

    # Thread functions for concurrent querying
    def query_openai():
        nonlocal openai_reply
        openai_reply = get_openai_response(question)

    def query_perplexity():
        nonlocal perplexity_reply
        perplexity_reply = get_perplexity_response(question)

    def query_langchain():
        nonlocal langchain_reply
        langchain_reply = run_langchain_query(question)

def fetch_tweets(hashtag):
    url = "https://real-time-x-com-data-scraper.p.rapidapi.com/Search/"
    querystring = {"q": f"#{hashtag}", "safe_search": "true", "count": "20", "language": "en"}
    headers = {
        "X-RapidAPI-Key": os.getenv('RAPIDAPI_KEY'),  # Ensure this key is in your .env file
        "X-RapidAPI-Host": "real-time-x-com-data-scraper.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        logger.info("Twitter API Response Data: %s", data)  # Debugging statement

        posts = [post.get('content', 'No content available') for post in data.get('posts', [])]
        return posts
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return []




def fetch_nasdaq_data():
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        raise ValueError("NASDAQ_DATA_LINK_API_KEY not found in environment variables")
    # Use the API key to fetch data

def count_stocks_with_ema_condition():
    """
    Count stocks with 12-day EMA below 26-day EMA in the last 7 days.
    """
    query = """
        SELECT COUNT(*) AS stock_count
        FROM stock_data
        WHERE ema_12 < ema_26
          AND date >= DATE('now', '-7 days');
    """
    try:
        conn, cursor = connect_to_database()
        cursor.execute(query)
        result = cursor.fetchone()
        return f"{result[0]} stocks traded with the 12-day EMA below the 26-day EMA in the last week."
    except Exception as e:
        return f"Error fetching data: {str(e)}"
    finally:
        if conn:
            conn.close()


def analyze_stock_with_langchain(symbol):
    db_query = """
        SELECT date, symbol, close_price, ema_12, ema_26, net, trend, extreme, sector, industry
        FROM stock_data
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 25
    """
    results = fetch_database_results(db_query, (symbol,))
    if isinstance(results, str):  # Error case
        return {"error": results}
    elif not results:  # No data case
        return {"error": f"No data found for symbol: {symbol}"}

    # Format results for LangChain
    rows = "\n".join(", ".join(str(item) for item in row) for row in results)
    prompt = f"Analyze the following stock data for {symbol}:\n{rows}"

    # LangChain analysis
    try:
        response = handle_langchain_query(prompt)
        return {"success": response}
    except Exception as e:
        logger.error(f"LangChain analysis error: {e}")
        return {"error": "LangChain analysis failed."}

def fetch_stock_data(symbol):
    # Replace with actual data retrieval logic
    # Example: Query your database for the stock data
    stock = Stock.objects.filter(symbol=symbol).first()
    if stock:
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "price": stock.price,
            "change": stock.change,
            "percent_change": stock.percent_change,
            # Include other relevant fields
        }
    return None



def handle_slash_command(command):
    if command.startswith("/google "):
        prompt = command.replace("/google ", "").strip()
        response = call_google_api(prompt)
        return {"reply": response.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "No response")}
    # Other commands...
    else:
        return {"reply": "Command not recognized."}

def handle_slash_command(command):
    if command.startswith('/ask-all:'):
        question = command.split(':', 1)[1].strip()
        return ask_all_apis(question)
    elif command == "/test-db":
        return test_database_connection()
    elif command.startswith("/run-chain-db:"):
        symbol = command[len("/run-chain-db:"):].strip()
        return run_langchain_analysis(symbol)
    elif command.startswith("/trend-range"):
        try:
            _, min_val, max_val = command.split()
            return get_trend_range(min_val, max_val)
        except ValueError:
            return JsonResponse({"error": "Invalid trend range format. Use: /trend-range MIN MAX"})

    elif command.startswith("/sector:"):
        sector = command[len("/sector:"):].strip()
        return get_stocks_by_sector(sector)
    elif command == "/emaB0-1":
        return get_ema_difference()
    # Add other commands here
    else:
        return "Command not recognized. Available commands: \n" \
               "- `/test-db`: Test database connection.\n" \
               "- `/run-chain-db:[symbol]`: Use LangChain to analyze stock data.\n" \
               "- `/trend-range MIN MAX`: Stocks with trends in the range.\n" \
               "- `/sector:[sector]`: List stocks in a sector.\n" \
               "- `/emaB0-1`: Stocks with EMA difference between 0% and 1%.\n" \
               "- `/ask-all:[question]`: Ask predefined set of APIs a question.\n" \
               "- `/google:[query]`: Direct Google API search."


def test_database_connection(_):
    try:
        conn = sqlite3.connect(os.getenv("DATABASE_PATH"))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return f"Database Tables: {', '.join(row[0] for row in tables)}"
    except sqlite3.Error as e:
        return f"Database connection error: {e}"

def handle_langchain_query(symbol):
    prompt = f"Analyze the following stock data for {symbol}."
    try:
        llm = OpenAI(temperature=0.7)
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["query"],
                template="Answer the following question: {query}",
            ),
        )
        return chain.run(prompt)
    except Exception as e:
        return f"LangChain Error: {e}"

def fetch_trend_range(range_args):
    try:
        min_val, max_val = map(float, range_args.split())
        # Example database query
        query = """
            SELECT symbol, trend
            FROM stock_data
            WHERE trend BETWEEN ? AND ?
            ORDER BY trend DESC
        """
        results = fetch_database_results(query, (min_val, max_val))
        if not results:
            return "No data found for the specified trend range."
        return "\n".join(f"{row[0]}: {row[1]}%" for row in results)
    except ValueError:
        return "Invalid trend range format. Use: `/trend-range MIN MAX`."

def get_stocks_by_sector(sector):
    """
    Fetch stocks in a specific sector and format as a table.
    """
    conn = connect_to_database()
    if not conn:
        return "Database connection failed."

    query = """
        SELECT symbol, sector, industry
        FROM stock_data
        WHERE sector LIKE ?
        LIMIT 25
    """
    try:
        with conn:
            cursor = conn.execute(query, (f"%{sector}%",))
            results = cursor.fetchall()
            if results:
                # Format the results into a table
                return format_data_as_table(results, headers=["Symbol", "Sector", "Industry"])
            else:
                return f"No data found for sector: {sector}"
    except sqlite3.Error as e:
        logger.error(f"Database query error: {e}")
        return "An error occurred while fetching data."

def format_data_simple(data, headers):
    """
    Formats a list of tuples into a simple string table.
    """
    # Create the header row
    table = f"{' | '.join(headers)}\n"
    table += f"{'-' * len(table)}\n"

    # Add data rows
    for row in data:
        table += f"{' | '.join(map(str, row))}\n"

    return table


def fetch_ema_difference(_):
    query = """
        SELECT symbol, ema_12, ema_26
        FROM stock_data
        WHERE ABS((ema_12 - ema_26) / ema_26 * 100) BETWEEN 0 AND 1
        LIMIT 10
    """
    results = fetch_database_results(query, ())
    if not results:
        return "No stocks found with EMA difference between 0% and 1%."
    return "\n".join(f"{row[0]}: EMA 12 = {row[1]}, EMA 26 = {row[2]}" for row in results)

def get_openai_response(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}],
            max_tokens=150,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenAI Error: {e}"

def get_perplexity_response(question):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    payload = {"question": question}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("answer", "No answer received.")
    except requests.exceptions.RequestException as e:
        return f"Perplexity Error: {e}"


def help_command():
    commands = [
        "/test-db: Test database connection.",
        "/run-chain-db:[symbol]: Use LangChain to analyze stock data.",
        "/trend-range MIN MAX: Stocks with trends in the range.",
        "/sector:[sector]: List stocks in a sector.",
        "/emaB0-1: Stocks with EMA difference between 0% and 1%.",
        "/[symbol]: Retrieve data for a specific stock symbol.",
        "Ask:[question]: Ask OpenAI a question.",
    ]
    help_text = "Available commands:\n" + "\n".join(commands)
    return JsonResponse({"response": help_text})


def handle_slash_command(command):

    # Handle /nasdaq command
    if command.startswith("/nasdaq:"):
        symbol = command.split(":")[1].strip().upper()
        if not symbol:
            return {"error": "Please provide a stock symbol after /nasdaq:."}
        data = fetch_nasdaq_data(symbol)
        if "error" in data:
            return {"reply": data["error"], "api_used": "Nasdaq API"}
        stock_data = data.get("dataset", {})
        name = stock_data.get("name", "N/A")
        description = stock_data.get("description", "N/A")
        latest_data = stock_data.get("data", [])[0]  # Assuming the first entry is the latest
        response = f"Stock: {name}\nDescription: {description}\nLatest Data: {latest_data}"
        return {"reply": response, "api_used": "Nasdaq API"}
    
    # Handle /compare command
    elif command.startswith("/compare"):
        return compare_stocks(command)
    

    
    # Handle /test-db command
    elif command.startswith("/test-db"):
        return test_database_connection()
    
    # Handle /run-chain-db command
    elif command.startswith("/run-chain-db"):
        return run_chain_analysis(command)
    
    # Handle /trend-range command
    elif command.startswith("/trend-range"):
        try:
            _, min_val, max_val = command.split()
            return get_trend_range(min_val, max_val)
        except ValueError:
            return {"error": "Invalid trend range format. Use: /trend-range MIN MAX"}
    
    # Handle /sector command
    elif command.startswith("/sector:"):
        sector = command.split(":", 1)[1].strip()
        return get_stocks_by_sector(sector)
    
    # Handle /emaB0-1 command
    elif command.startswith("/emaB0-1"):
        return get_ema_difference()
    
    # Handle Ask command
    elif command.startswith("Ask:"):
        question = command.split(":", 1)[1].strip()
        return get_openai_response(question)
    
    # Handle /ask-all command
    elif command.startswith('/ask-all:'):
        question = command.split(':', 1)[1].strip()
        return ask_all_apis(question)
    
    else:
        return {"reply": "Command not recognized. Available commands:\n" \
               "- `/test-db`: Test database connection.\n" \
               "- `/run-chain-db:[symbol]`: Use LangChain to analyze stock data.\n" \
               "- `/trend-range MIN MAX`: Stocks with trends in the range.\n" \
               "- `/sector:[sector]`: List stocks in a sector.\n" \
               "- `/emaB0-1`: Stocks with EMA difference between 0% and 1%.\n" \
               "- `Ask:[question]`: Ask OpenAI a question.\n" \
               "- `/ask-all:[question]`: Query OpenAI, Perplexity, and LangChain APIs simultaneously.",
               "api_used": "N/A"}
    if command.startswith('/ask-all:'):
        question = command.split(':', 1)[1].strip()
        return ask_all_apis(question)
    elif command.startswith('/test-db'):
        return test_database_connection()
    elif command.startswith('/run-chain-db:'):
        symbol = command.split(':', 1)[1].strip()
        return run_chain_db_analysis(symbol)
    elif command.startswith('/trend-range'):
        params = command.split(' ')[1:]
        if len(params) == 2:
            try:
                min_val, max_val = map(float, params)
                return fetch_trend_range(min_val, max_val)
            except ValueError:
                return "Invalid range. Use `/trend-range MIN MAX`."
        else:
            return "Invalid format. Use `/trend-range MIN MAX`."
    elif command.startswith("/google "):
        prompt = command.replace("/google ", "").strip()
        response = call_google_api(prompt)
        return {"reply": response.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "No response")}    
    
    elif command.startswith('/sector:'):
        sector = command.split(':', 1)[1].strip()
        return fetch_sector_data(sector)
    elif command.startswith('/emaB0-1'):
        return fetch_ema_bands(0, 1)
    elif command.startswith('Ask:'):
        question = command.split(':', 1)[1].strip()
        return get_openai_response(question)
    else:
        return "Command not recognized. Available commands:\n" \
               "- `/test-db`: Test database connection.\n" \
               "- `/run-chain-db:[symbol]`: Use LangChain to analyze stock data.\n" \
               "- `/trend-range MIN MAX`: Stocks with trends in the range.\n" \
               "- `/sector:[sector]`: List stocks in a sector.\n" \
               "- `/emaB0-1`: Stocks with EMA difference between 0% and 1%.\n" \
               "- `Ask:[question]`: Ask OpenAI a question.\n" \
               "- `/ask-all:[question]`: Query OpenAI, Perplexity, and LangChain APIs simultaneously."


def command_handler(request):
    """

    Args:
      request:

    Returns:

    """
    if request.method == "POST":
        user_input = request.POST.get("command", "").strip()
        if user_input.startswith("/google:"):
            return handle_google_command(user_input.split(":", 1)[1])
        elif user_input.startswith("/test-db"):
            return test_database_connection()
        else:
            return JsonResponse({"reply": "Command not recognized.", "api_used": "N/A"})

def command_handler(request):
    if request.method == "POST":
        user_input = request.POST.get("command", "").strip()
        
        if user_input.lower() == "/help":
            # Display a help message
            return help_command()


        
        elif user_input.startswith("/"):
            # Handle all slash commands (like `/ask-all`)
            return handle_slash_command(user_input)
        
        elif user_input.startswith("@"):
            # Handle all at commands (like `@sector`)
            return handle_at_command(user_input)
        
        else:
            # Invalid command format
            return JsonResponse({"error": "Invalid command format. Type '/help' for a list of valid commands."})
    
    return JsonResponse({"error": "Invalid request method. POST is required."})



def fetch_stock_data(stock_symbol):
    # Retrieve stock data from your database or an external API
    pass


def compare_stocks(data1, data2):
    # Compare the two datasets and return the result
    pass


def handle_at_command(command):
    if command.startswith("@fin-bearish"):
        return process_fin_bearish_command()
    # Add more at command handlers as needed
    else:
        return {"error": "Unknown at command."}


def process_stock_command(command):
    try:
        # Extract the stock symbol from the command
        stock_symbol = command[
            1:
        ].upper()  # Removes the leading '/' and converts to uppercase
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data:
            return {"stock_data": stock_data}
        else:
            return {"error": f"No data found for symbol {stock_symbol}."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def command_handler(request):
    if request.method == "POST":
        user_input = request.POST.get("command", "").strip()
        if user_input.startswith("/"):
            response = handle_slash_command(user_input)
        elif user_input.startswith("@"):
            response = handle_at_command(user_input)
        else:
            response = {"error": "Invalid command format."}
        return JsonResponse(response)
    return JsonResponse({"error": "Invalid request method."})


def fetch_latest_data():
    """
    Fetches the latest data for a specific symbol (e.g., 'AAPL') from the database.
    """
    conn = connect_to_database()
    if not conn:
        print("Failed to connect to the database.")
        return


def fetch_database_results(query, params=()):
    """
    Executes a query on the database and fetches results.
    """
    conn, cursor = connect_to_database()
    if not conn or not cursor:
        return "Error: Could not connect to the database."

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        return results if results else "No data found for the given query."
    except sqlite3.Error as e:
        logger.error(f"Database query error: {e}")
        return f"Database query error: {e}"
    finally:
        if conn:
            conn.close()


def get_perplexity_response(query):
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "No valid response received.")
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Perplexity API error: {e}")
        return f"Error: {e}"


def get_chatgpt_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specify the model to use
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            max_tokens=200,  # Adjust token limit as needed
            temperature=0.7,  # Set creativity level
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.OpenAIError as e:
        logger.error(f"ChatGPT API error: {e}")
        return "Error: Unable to fetch response from ChatGPT."



@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        user_message = request.POST.get("user_input", "").strip()

        # Route the command to the main handler
        response_text = main_command_handler(user_message)

        # Return the response
        return JsonResponse({"reply": response_text})
    return render(request, "chat_window.html")

def chat_view(request):
    """Handles GET (chat interface) and POST (command processing)."""
    if request.method == "GET":
        return render(
            request,
            "chat_window.html",
            {"current_time": datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")},
        )

    elif request.method == "POST":
        user_message = json.loads(request.body).get("message", "").strip()
        if not user_message:
            return JsonResponse({"reply": "No input received. Please type a message.", "api_used": "N/A"})

        # Example command processing
        if user_message == "/test-db":
            return JsonResponse({"reply": "Database connection is successful."})
        elif user_message.startswith("/echo"):
            return JsonResponse(
                {"reply": f"You said: {user_message.replace('/echo', '').strip()}"}
            )
        else:
            return JsonResponse(
                {"reply": "Command not recognized. Try /test-db or /echo [message]."}
            )



def handle_langchain_query(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Answer the following question: {query}",
    )
    llm = OpenAI(temperature=0.7)  # Ensure OpenAI is properly set up
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.run(query)
        return response
    except Exception as e:
        logger.error(f"LangChain error: {e}")
        return f"Error with LangChain: {e}"


@csrf_exempt
def chat_view(request):
    """Main chatbot view."""
    current_time = datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
    context = {
        "current_time": current_time,
        "reply": "Welcome to the chatbot! Please use the interface to enter a command.",
        "api_used": "N/A",
    }

    if request.method == "POST":
        user_message = (
            request.POST.get("user_input", "").strip()
            if request.content_type != "application/json"
            else json.loads(request.body).get("message", "").strip()
        )

        if not user_message:
            context["reply"] = "No input received. Please type a message."
            return JsonResponse(context)

        # Database Test Command
        if user_message.startswith("/test-db"):
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            results = fetch_database_results(query, ())
            if isinstance(results, str):
                context["reply"] = f"Database Error: {results}"
            else:
                context["reply"] = (
                    f"Tables in the database: {', '.join(row[0] for row in results)}"
                )
            context["api_used"] = "Database Test"

        # LangChain Command for Stock Analysis
        elif user_message.startswith("/run-chain-db:"):
            symbol = user_message.replace("/run-chain-db:", "").strip()
            query = """
                SELECT date, symbol, close_price, ema_12, ema_26, net, trend, extreme, sector, industry
                FROM stock_data
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 25
            """
            results = fetch_database_results(query, (symbol,))
            if isinstance(results, str):
                context["reply"] = results
            else:
                rows = "\n".join(
                    ", ".join(str(item) for item in row) for row in results
                )
                prompt = f"Summarize the following stock data for {symbol}:\n{rows}"
                context["reply"] = handle_langchain_query(prompt)
                context["api_used"] = "LangChain with Database"

        # Sector-Based Commands (e.g., @ commands)
        elif user_message.startswith("@"):
            # Define sector mapping for commands
            sector_map = {
                "@fin-bearish": "Financial Services",
                "@tech-bullish": "Technology",
                "@health-bearish": "Healthcare",
                "@energy-bullish": "Energy",
                "@real-bullish": "Real Estate",
                "@comm-bullish": "Communication Services",
                "@consdis-bearish": "Consumer Discretionary",
                "@consstap-bearish": "Consumer Staples",
                "@indus-bullish": "Industrials",
                "@util-bearish": "Utilities",
                "@mat-bullish": "Materials",
            }

            command = user_message.strip()
            sector = sector_map.get(command)
            if sector:
                # Determine bullish or bearish trend
                if "bearish" in command:
                    db_query = """
                        SELECT symbol, trend, sector, industry
                        FROM stock_data
                        WHERE sector = ? AND trend < 0
                        ORDER BY trend ASC
                        LIMIT 25
                    """
                else:
                    db_query = """
                        SELECT symbol, trend, sector, industry
                        FROM stock_data
                        WHERE sector = ? AND trend > 0
                        ORDER BY trend DESC
                        LIMIT 25
                    """
                results = fetch_database_results(db_query, (sector,))
                if isinstance(results, str):
                    context["reply"] = results
                else:
                    rows = "\n".join(
                        ", ".join(str(item) for item in row) for row in results
                    )
                    trend_type = "bearish" if "bearish" in command else "bullish"
                    context["reply"] = (
                        f"Top {trend_type} stocks in the {sector} sector:\n{rows}"
                    )
                    context["api_used"] = "Database Query"
            else:
                context["reply"] = (
                    "Sector command not recognized. Try commands like:\n"
                    "- `@fin-bearish` for Financial Services bearish stocks.\n"
                    "- `@tech-bullish` for Technology bullish stocks."
                )

        # Fetch Stocks by Trend Range
        elif user_message.startswith("/trend-range"):
            try:
                _, min_trend, max_trend = user_message.split()
                query = """
                    SELECT symbol, trend
                    FROM stock_data
                    WHERE trend BETWEEN ? AND ?
                    ORDER BY trend DESC
                """
                results = fetch_database_results(
                    query, (float(min_trend), float(max_trend))
                )
                if isinstance(results, str):
                    context["reply"] = results
                else:
                    rows = "\n".join(f"{row[0]}: {row[1]}%" for row in results)
                    context["reply"] = (
                        f"Stocks with trends between {min_trend}% and {max_trend}%:\n{rows}"
                    )
                    context["api_used"] = "Database Query"
            except ValueError:
                context["reply"] = (
                    "Invalid trend range format. Use: `/trend-range MIN MAX`."
                )

        # Fetch EMA-Based Stocks
        elif user_message.startswith("/emaB0-1"):
            query = """
                SELECT symbol, ema_12, ema_26
                FROM stock_data
                WHERE ABS((ema_12 - ema_26) / ema_26 * 100) BETWEEN 0 AND 1
                LIMIT 25
            """
            results = fetch_database_results(query, ())
            if isinstance(results, str):
                context["reply"] = results
            else:
                rows = "\n".join(
                    ", ".join(str(item) for item in row) for row in results
                )
                context["reply"] = (
                    f"Stocks with EMA differences between 0% and 1%:\n{rows}"
                )
                context["api_used"] = "Database Query"

        # OpenAI Command
        elif user_message.startswith("Ask:"):
            query = user_message.replace("Ask:", "").strip()
            context["reply"] = get_chatgpt_response(query)
            context["api_used"] = "OpenAI API"

        # Handle Invalid Commands
        else:
            context["reply"] = (
                "Command not recognized. Available commands:\n"
                "- `/test-db`: Test database connection.\n"
                "- `/run-chain-db:[symbol]`: Use LangChain to analyze stock data.\n"
                "- `/trend-range MIN MAX`: Stocks with trends in the range.\n"
                "- `/sector:[sector]`: List stocks in a sector.\n"
                "- `/emaB0-1`: Stocks with EMA difference between 0% and 1%.\n"
                "- `Ask:[question]`: Ask OpenAI a question.\n"
            )
            context["api_used"] = "N/A"

        # Save conversation history
        conversation_history = request.session.get("conversation_history", [])
        conversation_history.append({"user": user_message, "bot": context["reply"]})
        request.session["conversation_history"] = conversation_history

        return JsonResponse(context)


@csrf_exempt
def test_db_connection(request):
    """
    Test database connection and output available tables.
    """
    conn, cursor = connect_to_database()
    if not conn or not cursor:
        return JsonResponse(
            {"status": "error", "message": "Database connection failed."}
        )

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return JsonResponse({"status": "success", "tables": tables})
    except sqlite3.Error as e:
        return JsonResponse({"status": "error", "message": str(e)})
    finally:
        conn.close()


@csrf_exempt
def chat_view(request):
    """Main chatbot view."""
    current_time = datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
    context = {
        "current_time": current_time,
        "reply": "Welcome to the chatbot! Please use the interface to enter a command.",
        "api_used": "N/A",
    }

    if request.method == "POST":
        user_message = (
            request.POST.get("user_input", "").strip()
            if request.content_type != "application/json"
            else json.loads(request.body).get("message", "").strip()
        )

        if not user_message:
            context["reply"] = "No input received. Please type a message."
            return JsonResponse(context)

        # Database Test Command
        if user_message.startswith("/test-db"):
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            results = fetch_database_results(query, ())
            if isinstance(results, str):
                context["reply"] = f"Database Error: {results}"
            else:
                context["reply"] = (
                    f"Tables in the database: {', '.join(row[0] for row in results)}"
                )
            context["api_used"] = "Database Test"

        # LangChain Command for Stock Analysis
        elif user_message.startswith("/run-chain-db:"):
            symbol = user_message.replace("/run-chain-db:", "").strip()
            query = """
                SELECT date, symbol, close_price, ema_12, ema_26, net, trend, extreme, sector, industry
                FROM stock_data
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 25
            """
            results = fetch_database_results(query, (symbol,))
            if isinstance(results, str):
                context["reply"] = results
            else:
                rows = "\n".join(
                    ", ".join(str(item) for item in row) for row in results
                )
                prompt = f"Summarize the following stock data for {symbol}:\n{rows}"
                context["reply"] = handle_langchain_query(prompt)
                context["api_used"] = "LangChain with Database"

        # Fetch Stocks by Trend Range
        elif user_message.startswith("/trend-range"):
            try:
                _, min_trend, max_trend = user_message.split()
                query = """
                    SELECT symbol, trend
                    FROM stock_data
                    WHERE trend BETWEEN ? AND ?
                    ORDER BY trend DESC
                """
                results = fetch_database_results(
                    query, (float(min_trend), float(max_trend))
                )
                if isinstance(results, str):
                    context["reply"] = results
                else:
                    rows = "\n".join(f"{row[0]}: {row[1]}%" for row in results)
                    context["reply"] = (
                        f"Stocks with trends between {min_trend}% and {max_trend}%:\n{rows}"
                    )
                    context["api_used"] = "Database Query"
            except ValueError:
                context["reply"] = (
                    "Invalid trend range format. Use: `/trend-range MIN MAX`."
                )

        # Fetch Sector-Based Stocks
        elif user_message.startswith("/sector:"):
            sector = user_message.replace("/sector:", "").strip()
            query = """
                SELECT symbol, sector, industry
                FROM stock_data
                WHERE sector LIKE ?
                LIMIT 25
            """
            results = fetch_database_results(query, (f"%{sector}%",))
            if isinstance(results, str):
                context["reply"] = results
            else:
                rows = "\n".join(
                    ", ".join(str(item) for item in row) for row in results
                )
                context["reply"] = f"Top companies in {sector} sector:\n{rows}"
                context["api_used"] = "Database Query"

        # Fetch EMA-Based Stocks
        elif user_message.startswith("/emaB0-1"):
            query = """
                SELECT symbol, ema_12, ema_26
                FROM stock_data
                WHERE ABS((ema_12 - ema_26) / ema_26 * 100) BETWEEN 0 AND 1
                LIMIT 10
            """
            results = fetch_database_results(query, ())
            if isinstance(results, str):
                context["reply"] = results
            else:
                rows = "\n".join(
                    ", ".join(str(item) for item in row) for row in results
                )
                context["reply"] = (
                    f"Stocks with EMA differences between 0% and 1%:\n{rows}"
                )
                context["api_used"] = "Database Query"

        # OpenAI Command
        elif user_message.startswith("Ask:"):
            query = user_message.replace("Ask:", "").strip()
            context["reply"] = get_chatgpt_response(query)
            context["api_used"] = "OpenAI API"

        # Handle Invalid Commands
        else:
            context["reply"] = (
                "Command not recognized. Available commands:\n"
                "- `/test-db`: Test database connection.\n"
                "- `/run-chain-db:[symbol]`: Use LangChain to analyze stock data.\n"
                "- `/trend-range MIN MAX`: Stocks with trends in the range.\n"
                "- `/sector:[sector]`: List stocks in a sector.\n"
                "- `/emaB0-1`: Stocks with EMA difference between 0% and 1%.\n"
                "- `Ask:[question]`: Ask OpenAI a question.\n"
            )
            context["api_used"] = "N/A"

        # Save conversation history
        conversation_history = request.session.get("conversation_history", [])
        conversation_history.append({"user": user_message, "bot": context["reply"]})
        request.session["conversation_history"] = conversation_history

        return JsonResponse(context)

    # Render Chat Interface for GET Requests
    return render(request, "chat_window.html", context)
