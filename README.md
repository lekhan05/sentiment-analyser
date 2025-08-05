# sentiment-analyser
A simple sentiment analyser of user reviews using LLM (llama exposed via groq) and langchain. 
 1. Gives a summary of the user review.
 2. Categorises user review as postivie, negative or neutral.
 3. Generates appropriate reply to the user review based on the sentiment.
# Steps
1. Install UV package manager
2. Create a venv
3. Perform a uv sync to download the dependencies
4. Create an API key for accessing groq (https://console.groq.com/playground)
5. To run the snetiment analyser: uv run sentiment_analyser.py
