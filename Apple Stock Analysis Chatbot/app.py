# Importing necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import gradio as gr
import google.generativeai as genai
import re
from word2number import w2n

# Fetch stock data with a default period
def fetch_data_yahoo(ticker: str, max_period: str = "max"):
    # Define valid periods supported by yfinance
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    
    # Validate the max_period parameter
    if max_period not in valid_periods:
        raise ValueError(
            f"Invalid period specified. Due to limitations in Yahoo Finance's data API, "
            f"the supported periods are: {', '.join(valid_periods)}."
        )
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=max_period)
        
        # Convert all timestamps to naive (no timezone)
        data.index = data.index.tz_localize(None)
        
        if data.empty:
            raise ValueError(f"No data fetched for {ticker}. Check the ticker symbol or availability of data.")
        return data
    except Exception as e:
        print(f"Error occurred while fetching data: {e}")
        return None


# Preprocess data
def validate_and_preprocess_data(data, max_period):
    if data is None or data.empty:
        return None
    if max_period in ['5y', '10y'] and len(data) < 200:  # Check if data is too short for long periods
        return None
    data = data.ffill()  # Fill missing values
    return data

# Plot stock data
def plot_stock_data(data):
    if data is None or data.empty:
        return None
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label="Close Price", color='blue')
    plt.title("Stock Closing Price")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('stock_chart.png')
    plt.close()
    return 'stock_chart.png'

# Configure the Gemini API key
genai.configure(api_key="YOUR_API_KEY")  

# LangChain prompt for stock performance
def generate_summary(data, ticker, period):
    if data is None or data.empty:
        return "No data available to summarize."
    
    # Include all data in the prompt
    stock_data_str = data[['Close']].to_string()
    
    prompt = f"""
    Summarize the stock performance of {ticker} over the last {period}.
    The data provided includes closing prices for the selected period. Summarize major trends and key takeaways.
    
    Data: {stock_data_str}
    """
    
    try:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 500,
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary."

# Enhanced input parser to handle natural language
def parse_user_input(input_text):
    ticker = "AAPL"  # Default ticker
    max_period = "max"  # Default max period (changed to 'max')

    # Extract ticker symbol
    ticker_match = re.search(r"\b[A-Z]{1,5}\b", input_text)
    if ticker_match:
        ticker = ticker_match.group(0)
    
    # Extract period (convert words to numbers if necessary)
    period_match = re.search(r"(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)\s*(year|month|day)s?", input_text, re.IGNORECASE)
    if period_match:
        period_value = period_match.group(1)
        unit = period_match.group(2).lower()

        # Convert word numbers to digits if necessary
        try:
            if period_value.isalpha():
                period_value = str(w2n.word_to_num(period_value.lower()))  # Convert word to number
        except ValueError:
            pass  # If it's not a valid word, just leave it as is
        
        num = int(period_value)

        # Validation based on units and periods
        if unit == "year" and num > 10:
            return ticker, None, "I only have data for up to 10 years."
        elif unit == "month" and num > 120:
            return ticker, None, "I only have data for up to 120 months."
        elif unit == "day" and num > 3650:
            return ticker, None, "I only have data for up to 3650 days."
        else:
            unit_abbr = {"year": "y", "month": "mo", "day": "d"}[unit]
            max_period = f"{num}{unit_abbr}"
    
    return ticker, max_period, None

# Gradio function to process user queries
def analyze_stock(user_input):
    ticker, max_period, error_message = parse_user_input(user_input)
    if error_message:
        return error_message, None
    
    if ticker not in ["AAPL"]:  # Check if the ticker is supported
        return "Sorry, I currently support data only for Apple (AAPL) stocks.", None

    try:
        data = fetch_data_yahoo(ticker, max_period)
        if data is not None:
            data = validate_and_preprocess_data(data, max_period)
            if data is None:
                return f"The data for {max_period} is insufficient. Please try a shorter period.", None
            
            summary = generate_summary(data, ticker, max_period)
            plot_path = plot_stock_data(data)
            return summary, plot_path
        else:
            return "No data available for this stock ticker.", None
    except ValueError as ve:
        return str(ve), None
    except Exception as e:
        return f"Error occurred: {e}", None

# Gradio interface
interface = gr.Interface(
    fn=analyze_stock,
    inputs=gr.Textbox(label="Enter Query:"),
    outputs=[
        gr.Textbox(label="Generated Summary"),
        gr.Image(label="Stock Price Chart")
    ],
    title="Apple Stock Analysis Tool",
    description="Ask about stock performance in any way. For example: 'How did AAPL stocks perform over the last 5 years?'"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)