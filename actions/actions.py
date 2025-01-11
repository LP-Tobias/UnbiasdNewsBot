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
from rasa_sdk.events import ActionExecuted, SlotSet, AllSlotsReset

import openai
from newsapi import NewsApiClient

from Dbias.text_debiasing import *
from Dbias.bias_classification import *
from Dbias.bias_recognition import *
from Dbias.bias_masking import *

logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", "")

openai.api_key = OPENAI_API_KEY

class ActionFetchNews(Action):
    def name(self) -> Text:
        return "action_fetch_news"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        topic = tracker.get_slot("topic")
        n_news = tracker.get_slot("n_news")

        if not n_news:
            n_news = 1

        if not topic:
            dispatcher.utter_message(text="Please specify a topic for news.")
            return []

        n_news = int(n_news) if n_news else 1  # Default to 1

        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        try:
            response = newsapi.get_everything(
                q=topic,
                language="en",
                page_size=int(n_news)
            )
        except Exception as e:
            dispatcher.utter_message(text=f"Error fetching news: {e}")
            return []

        if response.get("status") != "ok":
            dispatcher.utter_message(text="I encountered an error fetching the news.")
            return []

        articles = response.get("articles", [])
        if not articles:
            dispatcher.utter_message(text="I didn't find relevant news on that topic.")
            return []

        dispatcher.utter_message(text=f"Here are the latest {len(articles)} articles I found:")

        # Collect articles as structured data
        structured_articles = []

        for idx, article in enumerate(articles, start=1):
            structured_articles.append({
                "title": article.get("title", "Untitled"),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "source": article.get("source", {}).get("name", "Unknown Source")
            })

            dispatcher.utter_message(
                text=(
                    f"**Article {idx}:**\n"
                    f"Title: {structured_articles[-1]['title']}\n"
                    f"Source: {structured_articles[-1]['source']}\n"
                    f"Description: {structured_articles[-1]['description']}\n"
                    f"Content: {structured_articles[-1]['content']}\n"
                )
            )

        # Store structured data as JSON in the slot
        return [{"event": "slot", "name": "news_fetched", "value": structured_articles}]


class ActionSummarizeNews(Action):
    def name(self) -> Text:
        return "action_summarize_news"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[Dict[Text, Any]]:

        news_articles = tracker.get_slot("news_fetched")
        if not news_articles:
            dispatcher.utter_message(text="I don't have news to summarize.")
            return []

        summaries = []
        for idx, article in enumerate(news_articles, start=1):
            content = article.get("content", "")

            if not content:
                continue

            # Summarize each article
            system_prompt = (
                "You are a professional news assistant. Your task is to generate a clear and concise summary "
                "of the following news article. Focus on the key facts, main ideas, and important takeaways that readers "
                "would naturally remember. Preserve the original tone and intent of the article without adding, removing, "
                "or altering any information. Do not perform any bias detection, analysis, or modifications."
            )

            user_prompt = f"{content}\n\nPlease provide a concise summary of this article."


            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            summary = response["choices"][0]["message"]["content"]
            summaries.append(summary)

            dispatcher.utter_message(text=f"**Summary of Article {idx}:**\n{summary}")

        return [{"event": "slot", "name": "news_summary", "value": summaries}]
    

class ActionBiasDetectionSummary(Action):
    """Detect bias in the summarized news content."""

    def name(self) -> Text:
        return "action_bias_detection_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        # Get the summarized news from the slot
        news_summary = tracker.get_slot("news_summary")
        if not news_summary:
            dispatcher.utter_message(text="I don't have any summarized news to analyze.")
            return []

        # Handle both single summary and list of summaries
        if isinstance(news_summary, str):
            summaries = [news_summary]
        else:
            summaries = news_summary

        bias_results = []

        for idx, summary in enumerate(summaries, start=1):
            # Detect bias using the classifier
            bias_result = classifier(summary)
            bias_results.append(bias_result)

            dispatcher.utter_message(
                text=f"**Bias Analysis of Summary {idx}:**\n{bias_result}"
            )

        # Store bias results in a slot
        return [{"event": "slot", "name": "summary_bias", "value": bias_results}]


class ActionBiasRecognitionSummary(Action):
    """Recognize biased words in the summarized news content."""

    def name(self) -> Text:
        return "action_bias_recognition_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        # Get the summarized news from the slot
        news_summary = tracker.get_slot("news_summary")
        if not news_summary:
            dispatcher.utter_message(text="I don't have any summarized news to analyze.")
            return []

        # Handle both single summary and list of summaries
        if isinstance(news_summary, str):
            summaries = [news_summary]
        else:
            summaries = news_summary

        biased_words_summary = []

        for idx, summary in enumerate(summaries, start=1):
            # Recognize biased words in the summary
            biased_words = recognizer(summary)
            biased_word_list = [word['entity'] for word in biased_words]

            # Provide feedback to the user
            if biased_word_list:
                biased_words_text = ", ".join(biased_word_list)
                dispatcher.utter_message(
                    text=f"**Biased Words in Summary {idx}:** {biased_words_text}"
                )
            else:
                dispatcher.utter_message(
                    text=f"**Summary {idx}:** No biased words detected."
                )

            biased_words_summary.append({
                "summary_index": idx,
                "biased_words": biased_word_list
            })

        # Store biased words in a slot
        return [{"event": "slot", "name": "summary_biased_words", "value": biased_words_summary}]


class ActionDbiasSummary(Action):
    """Debias the summarized news content (Method 1) using custom debiasing."""

    def name(self) -> Text:
        return "action_dbias_summary"
    
    def custom_debiasing(self, x):
        suggestions = run(x)
        if suggestions == None:
            return ""
        else:
            all_suggestions = []
            for sent in suggestions[0:3]:
                all_suggestions.append(sent['Sentence'])
            return "\n\n".join(all_suggestions)

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        # Retrieve the summarized news from the slot
        news_summary = tracker.get_slot("news_summary")

        if not news_summary:
            dispatcher.utter_message(text="I don't have any summarized news to debias.")
            return []

        # Handle both single summary and list of summaries
        if isinstance(news_summary, str):
            summaries = [news_summary]
        else:
            summaries = news_summary

        debiased_summaries = []

        for idx, summary in enumerate(summaries, start=1):
            # Apply custom debiasing to each summary
            debiased_text = self.custom_debiasing(summary)

            # Provide feedback to the user
            if debiased_text.strip():
                dispatcher.utter_message(
                    text=f"**Debiased Summary {idx}:**\n{debiased_text}"
                )
            else:
                dispatcher.utter_message(
                    text=f"**Summary {idx}:** No bias detected or no debiasing suggestions available."
                )

            debiased_summaries.append({
                "summary_index": idx,
                "debiased_text": debiased_text
            })

        # Store the debiased summaries in the 'summary_bias' slot
        return [{"event": "slot", "name": "Dbias_summary", "value": debiased_summaries}]
    

class ActionLlmDebiasSummary(Action):
    """Debias multiple summarized news articles using LLM with Chain-of-Thought reasoning."""

    def name(self) -> Text:
        return "action_llm_debias_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        # Retrieve the summarized news from the slot
        news_summaries = tracker.get_slot("news_summary")

        if not news_summaries:
            dispatcher.utter_message(text="I don't have any summaries to debias.")
            return []

        # Ensure summaries are in a list format
        if isinstance(news_summaries, str):
            summaries = [news_summaries]
        else:
            summaries = news_summaries

        debiased_summaries = []

        # Define the system prompt for debiasing
        system_prompt = (
            "You are a neutral AI assistant. Carefully analyze the following text for bias. "
            "If you find subjective, emotionally charged, or inflammatory language, rewrite it to be neutral and factual. "
            "Focus on preserving the original meaning while ensuring an unbiased tone."
        )

        for idx, summary in enumerate(summaries, start=1):
            user_prompt = f"{summary}\n\nRewrite the text to remove any potential biases or subjective language."

            try:
                # Call the LLM for each summary
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                debiased_text = response["choices"][0]["message"]["content"].strip()

                # Provide feedback to the user for each debiased summary
                dispatcher.utter_message(
                    text=f"**Debiased Summary {idx}:**\n{debiased_text}"
                )

                # Collect the debiased summary
                debiased_summaries.append({
                    "summary_index": idx,
                    "original_summary": summary,
                    "debiased_summary": debiased_text
                })

            except Exception as e:
                dispatcher.utter_message(
                    text=f"An error occurred while debiasing summary {idx}: {e}"
                )

        # Store all debiased summaries in a slot
        return [{"event": "slot", "name": "LLM_debias_summary", "value": debiased_summaries}]



class ActionLlmSummaryBiasDetection(Action):
    """Use Dbias package to detect biases in the debiased summaries after LLM processing."""

    def name(self) -> Text:
        return "action_llm_summary_bias_detection"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        # Retrieve the debiased summaries from the slot
        debiased_summaries = tracker.get_slot("LLM_debias_summary")

        if not debiased_summaries:
            dispatcher.utter_message(text="I don't have any debiased summaries to analyze.")
            return []

        # Ensure summaries are in a list format
        if isinstance(debiased_summaries, str):
            summaries = [debiased_summaries]
        else:
            summaries = debiased_summaries

        bias_results = []

        for idx, summary in enumerate(summaries, start=1):
            try:
                summary_text = summary.get('original_summary')
                # Detect bias using the classifier
                bias_result = classifier(summary_text)
                
                # Provide feedback to the user for each summary
                dispatcher.utter_message(
                    text=f"**Bias Detection for Summary {idx}:** {bias_result}"
                )

                # Collect the bias detection results
                bias_results.append({
                    "summary_index": idx,
                    "bias_classification": bias_result
                })

            except Exception as e:
                dispatcher.utter_message(
                    text=f"Error analyzing bias in Summary {idx}: {str(e)}"
                )

        # Store the bias detection results in a slot
        return [
            {"event": "slot", "name": "LLM_summary_score", "value": bias_results}
            # ActionExecuted("action_listen"),
            # SlotSet("news_summary", None),
            # SlotSet("summary_bias", None),
            # AllSlotsReset(),
            # ActionExecuted("action_listen")
        ]


# class ActionBiasDetection(Action):
#     """Use Dbias package to detect biases in the news content and print biased words."""

#     def name(self) -> Text:
#         return "action_bias_detection"
    
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: DomainDict) -> List[Dict[Text, Any]]:
        
#         # Get the news content
#         news_text = tracker.get_slot("news_fetched")
#         if not news_text:
#             dispatcher.utter_message(text="I don't have news to analyze.")
#             return []

#         try:
#             # Use the Dbias package to classify the bias
#             bias = classifier(news_text)
#             dispatcher.utter_message(text=f"The news content is classified as: {bias}")

#             # Recognize biased words in the news content
#             biased_words = recognizer(news_text)
#             if biased_words:
#                 biased_words_list = [word['entity'] for word in biased_words]
#                 biased_words_text = ", ".join(biased_words_list)
#                 dispatcher.utter_message(text=f"Identified biased words: {biased_words_text}")
#             else:
#                 dispatcher.utter_message(text="No biased words were detected.")
        
#         except Exception as e:
#             dispatcher.utter_message(text=f"An error occurred during bias detection: {e}")
#             return []

#         # Optionally, store the bias and biased words in slots
#         # return [
#         #     {"event": "slot", "name": "news_bias", "value": bias},
#         #     {"event": "slot", "name": "biased_words", "value": biased_words_text}
#         # ]
        
#         return []

class ActionBiasRecognition(Action):
    """Use Dbias package to recognize biases in the news content."""

    def name(self) -> Text:
        return "action_bias_recognition"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        
        news_text = tracker.get_slot("news_fetched")
        if not news_text:
            dispatcher.utter_message(text="I don't have news to analyze.")
            return []
        
        def custom_recognizer(x):
            biased_words = recognizer(x)
            biased_words_list = []
            for id in range(0, len(biased_words)):
                biased_words_list.append(biased_words[id]['entity'])
            return ", ".join(biased_words_list)
        
        biased_words_list_out = custom_recognizer(news_text)
        dispatcher.utter_message(text=f"These are the biased words: {biased_words_list_out}")
        
        # Use a slot to save the biased words list for LLM debias.
        return[]
        


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

        dispatcher.utter_message(text="❓ I'm sorry, I didn't understand that. Can you rephrase?")
        return [AllSlotsReset(), ActionExecuted("action_listen")]

class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="✅ I'm ready for your next request!")
        return [AllSlotsReset(), ActionExecuted("action_listen")]


