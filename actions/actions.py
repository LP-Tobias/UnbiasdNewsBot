# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# actions.py
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import logging
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

# Optional:
# from utils.rag_pipeline import retrieve_and_generate

# Make sure to `pip install langchain openai google-api-python-client requests` as needed.
import openai
from newsapi import NewsApiClient

from Dbias.bias_classification import classifier
from Dbias.bias_recognition import recognizer

logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", "")

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

class ActionFetchNews(Action):
    def name(self) -> Text:
        return "action_fetch_news"

    async def run(
        self, 
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        topic = tracker.get_slot("topic")
        if not topic:
            dispatcher.utter_message(text="Please specify a topic for news.")
            return []

        # Initialize the NewsAPI client
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        try:
            response = newsapi.get_everything(
                q=topic,
                language="en",
                page_size=1
            )
        except Exception as e:
            dispatcher.utter_message(text=f"Error fetching news: {e}")
            return []

        # Check if the API returned a successful response
        if response.get("status") != "ok":
            dispatcher.utter_message(text="I encountered an error fetching the news.")
            return []

        articles = response.get("articles", [])
        if not articles:
            dispatcher.utter_message(text="I didn't find relevant news on that topic.")
            return []

        # We'll build a combined text string AND also show each article to the user
        combined_news_text = ""
        dispatcher.utter_message(text="Here are the latest articles I found:")

        for idx, article in enumerate(articles, start=1):
            title = article.get("title", "Untitled")
            description = article.get("description", "")
            content = article.get("content", "")
            source_name = article.get("source", {}).get("name", "Unknown Source")

            # Send each article to the user
            dispatcher.utter_message(
                text=(
                    f"**Article {idx}:**\n"
                    f"Title: {title}\n"
                    f"Source: {source_name}\n"
                    f"Description: {description}\n"
                    f"Content: {content}\n"
                )
            )

            # Also build the combined text
            combined_news_text += (
                f"Title: {title}\n"
                f"Source: {source_name}\n"
                f"Description: {description}\n"
                f"Content: {content}\n\n"
            )

        # Store in a slot for subsequent summarization or debiasing
        return [
            {"event": "slot", "name": "news_summary", "value": combined_news_text}
        ]

class ActionBiasDetection(Action):
    """Use Dbias package to detect biases in the news content."""

    def name(self) -> Text:
        return "action_bias_detection"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        
        # Get the news content
        news_text = tracker.get_slot("news_summary")
        if not news_text:
            dispatcher.utter_message(text="I don't have news to analyze.")
            return []
        
        # Use the Dbias package to classify the bias
        bias = classifier(news_text)
        dispatcher.utter_message(text=f"The news content is classified as: {bias}")
        # probably store the bias in a slot for future reference
        # return [
        #     {"event": "slot", "name": "news_bias", "value": bias}
        # ]

class ActionBiasRecognition(Action):
    """Use Dbias package to recognize biases in the news content."""

    def name(self) -> Text:
        return "action_bias_recognition"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        
        pass

class ActionSummarizeNews(Action):
    """Use an LLM to summarize the fetched news."""

    def name(self) -> Text:
        return "action_summarize_news"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        news_text = tracker.get_slot("news_summary")
        if not news_text:
            dispatcher.utter_message(text="I don't have news to summarize.")
            return []
        
        # Basic prompt for summarization (Chain-of-Thought can be integrated here)
        system_prompt = (
            "You are a helpful news assistant. Summarize the following articles in a concise way: "
        )
        
        user_prompt = f"{news_text}\n\nSummarize the above content."

        # Call the LLM (OpenAI ChatGPT or any LLM)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        summary = response["choices"][0]["message"]["content"]
        dispatcher.utter_message(text="Here is a brief summary of the news:")
        dispatcher.utter_message(text=summary)
        
        return [
            {"event": "slot", "name": "news_summary", "value": summary}
        ]


class ActionDebiasSummary(Action):
    """Second pass to remove biases using Chain-of-Thought or a specialized prompt."""

    def name(self) -> Text:
        return "action_debias_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        current_summary = tracker.get_slot("news_summary")
        if not current_summary:
            dispatcher.utter_message(text="No summary found for debiasing.")
            return []
        
        # Example Chain-of-Thought or specialized prompt
        system_prompt = (
            "You are a neutral AI assistant. Check the text for bias. "
            "If you find any subjective or inflammatory language, rewrite it to be neutral."
        )
        user_prompt = f"{current_summary}\n\nRewrite the text to remove any potential biases or subjective language."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        debiased_summary = response["choices"][0]["message"]["content"]
        dispatcher.utter_message(text="Here is the debiased summary:")
        dispatcher.utter_message(text=debiased_summary)
        
        return [
            {"event": "slot", "name": "news_summary", "value": debiased_summary}
        ]


class ActionPerspectiveFeedback(Action):
    """Use Perspective API to evaluate potential toxicity or bias in the text."""

    def name(self) -> Text:
        return "action_perspective_feedback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        text_to_evaluate = tracker.get_slot("news_summary")
        if not text_to_evaluate:
            dispatcher.utter_message(text="No summary to evaluate.")
            return []
        
        url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        params = {
            "key": PERSPECTIVE_API_KEY
        }
        body = {
            "comment": {"text": text_to_evaluate},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}, "SEVERE_TOXICITY": {}, "INSULT": {}}
        }
        
        response = requests.post(url, params=params, json=body)
        
        if response.status_code != 200:
            dispatcher.utter_message(text="Error calling Perspective API.")
            return []
        
        data = response.json()
        attribute_scores = data.get("attributeScores", {})
        
        # Extract scores
        toxicity_score = attribute_scores.get("TOXICITY", {}).get("summaryScore", {}).get("value", 0)
        severe_toxicity_score = attribute_scores.get("SEVERE_TOXICITY", {}).get("summaryScore", {}).get("value", 0)
        insult_score = attribute_scores.get("INSULT", {}).get("summaryScore", {}).get("value", 0)
        
        # Provide feedback
        feedback_msg = (
            f"Perspective API scores:\n"
            f"- Toxicity: {toxicity_score}\n"
            f"- Severe Toxicity: {severe_toxicity_score}\n"
            f"- Insult: {insult_score}\n\n"
        )
        
        dispatcher.utter_message(text=feedback_msg)
        
        # Optionally prompt LLM to refine further if the scores are high
        if toxicity_score > 0.5 or severe_toxicity_score > 0.5 or insult_score > 0.5:
            refine_prompt = (
                "It seems there might be negative or biased language in the text. "
                "Please refine and ensure it is neutral and free from toxicity."
            )
            user_prompt = text_to_evaluate

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": refine_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            refined_text = response["choices"][0]["message"]["content"]
            dispatcher.utter_message(text="Refined text to reduce toxicity/bias:")
            dispatcher.utter_message(text=refined_text)
            return [
                {"event": "slot", "name": "news_summary", "value": refined_text}
            ]
        else:
            dispatcher.utter_message(text="The text appears neutral enough.")
            return []


class ActionDefaultFallback(Action):
    """Default fallback action if the user input can't be handled."""

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict
            ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_default")
        return []

