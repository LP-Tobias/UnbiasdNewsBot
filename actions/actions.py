# actions.py

from dotenv import load_dotenv
load_dotenv()

import os
import json
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

from newspaper import Article

from .topics import topic_list, subtopic_dict

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

        n_news = int(n_news)
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        try:
            response = newsapi.get_everything(q=topic, language="en", page_size=n_news)
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

        structured_articles = []

        for idx, article in enumerate(articles, start=1):
            url = article.get("url", "")
            full_content = "Content not available."

            if url:
                try:
                    news_article = Article(url)
                    news_article.download()
                    news_article.parse()
                    full_content = news_article.text
                except Exception as e:
                    full_content = "Unable to fetch full content."
                    print(f"Error fetching full content for {url}: {e}")

            structured_articles.append({
                "title": article.get("title", "Untitled"),
                "description": article.get("description", ""),
                "content": full_content,
                "source": article.get("source", {}).get("name", "Unknown Source"),
                "url": url
            })

            dispatcher.utter_message(
                text=(
                    f"**Article {idx}:**\n"
                    f"Title: {structured_articles[-1]['title']}\n"
                    f"Source: {structured_articles[-1]['source']}\n"
                    f"Description: {structured_articles[-1]['description']}\n"
                    f"Full Content: {structured_articles[-1]['content'][:500]}...\n"
                    f"Read more: {url}\n"
                )
            )

        return [{"event": "slot", "name": "news_fetched", "value": structured_articles}]

"""
# Back up incase newspaper3k stop working.

# class ActionFetchNews(Action):
#     def name(self) -> Text:
#         return "action_fetch_news"

#     async def run(
#         self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
#     ) -> List[Dict[Text, Any]]:

#         topic = tracker.get_slot("topic")
#         n_news = tracker.get_slot("n_news")

#         if not n_news:
#             n_news = 1

#         if not topic:
#             dispatcher.utter_message(text="Please specify a topic for news.")
#             return []

#         n_news = int(n_news) if n_news else 1  # Default to 1

#         # Initialize NewsAPI client
#         newsapi = NewsApiClient(api_key=NEWS_API_KEY)

#         try:
#             response = newsapi.get_everything(
#                 q=topic,
#                 language="en",
#                 page_size=int(n_news)
#             )
#         except Exception as e:
#             dispatcher.utter_message(text=f"Error fetching news: {e}")
#             return []

#         if response.get("status") != "ok":
#             dispatcher.utter_message(text="I encountered an error fetching the news.")
#             return []

#         articles = response.get("articles", [])
#         if not articles:
#             dispatcher.utter_message(text="I didn't find relevant news on that topic.")
#             return []

#         dispatcher.utter_message(text=f"Here are the latest {len(articles)} articles I found:")

#         # Collect articles as structured data
#         structured_articles = []

#         #TODO: In the distribution product, hide the content, provide a link to the article

#         for idx, article in enumerate(articles, start=1):
#             structured_articles.append({
#                 "title": article.get("title", "Untitled"),
#                 "description": article.get("description", ""),
#                 "content": article.get("content", ""),
#                 "source": article.get("source", {}).get("name", "Unknown Source")
#             })

#             dispatcher.utter_message(
#                 text=(
#                     f"**Article {idx}:**\n"
#                     f"Title: {structured_articles[-1]['title']}\n"
#                     f"Source: {structured_articles[-1]['source']}\n"
#                     f"Description: {structured_articles[-1]['description']}\n"
#                     f"Content: {structured_articles[-1]['content']}\n"
#                 )
#             )

#         # Store structured data as JSON in the slot
#         return [{"event": "slot", "name": "news_fetched", "value": structured_articles}]
"""

class ActionSummarizeNews(Action):
    """Summarize the fetched news articles using GPT-3.5-turbo. Remain factual."""

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

            dispatcher.utter_message(text=f"**Here is GPT3.5's Concise Summary of Article {idx}:**\n{summary}")

        return [{"event": "slot", "name": "news_summary", "value": summaries}]
    

class ActionBiasDetectionSummary(Action):
    """Dbias classification score in the summarized news content."""

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
                text=f"**Bias Score of Concise Summary {idx}:**\n{bias_result}"
            )

        # Store bias results in a slot
        return [{"event": "slot", "name": "summary_bias", "value": bias_results}]


class ActionBiasRecognitionSummary(Action):
    """Dbias recognize biased words in the summarized news content."""
    #TODO: The detected word can be further intriduced into the final LLM debiasing process.

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
    """Debias the summarized news content (Method 1) using method in Dbias."""

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
                    text=f"**Debiased Summary processed by Dbias {idx}:**\n{debiased_text}"
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
    """Debias multiple summarized news articles using LLM with prompts. (Method 2) """

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
                    text=f"**Single LLM Prompting Debiased Summary {idx}:**\n{debiased_text}"
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
                summary_text = summary.get('debiased_summary')
                # Detect bias using the classifier
                bias_result = classifier(summary_text)
                
                # Provide feedback to the user for each summary
                dispatcher.utter_message(
                    text=f"**Bias Score for Single-LLM-Prompting Summary {idx}:**\n{bias_result}"
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
        ]
    

class ActionLlmAnalysis(Action):
    """This comes after News Fetch, refer to mediabiasdetector"""

    def name(self) -> Text:
        return "action_llm_analysis"
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[Dict[Text, Any]]:

        news_articles = tracker.get_slot("news_fetched")
        if not news_articles:
            dispatcher.utter_message(text="I don't have news to summarize.")
            return []

        # Helper function to call OpenAI ChatCompletion
        def openai_chat(prompt: str, model: str = "gpt-4o") -> Dict:
            """
            Sends the prompt to OpenAI ChatCompletion and returns a parsed dictionary of the JSON response.
            NOTE: We assume the LLM's response is strictly valid JSON. 
                  You may want additional error handling if the response is not well-formed JSON.
            """
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. "
                                "The user will provide a news article and instructions to return only valid JSON. "
                                "Follow the instructions exactly."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0
                )
                response_text = response["choices"][0]["message"]["content"].strip()
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return an empty dict or handle as needed
                return {}
            except Exception as e:
                # Generic error handling
                print(f"OpenAI API call error: {e}")
                return {}

        # We'll store results in a list of dictionaries, one dict per article
        analysis_results = []

        for article_data in news_articles:
            # We'll assume each item has 'title' and 'body'
            article_title = article_data.get("title", "")
            article_body = article_data.get("content", "")

            # 1) Analyze the takeaways
            takeaways_prompt = f"""
            The following is a news article. Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_body}

            ####################

            Task: Summarize the main points of the news article.

            Instruction: List short takeaway points that readers are likely to remember from the article.
            Key: "takeaways"
            Value: A 3-4 sentence summary.
            """
            parsed_takeaways = openai_chat(takeaways_prompt, model="gpt-4o")

            # 2) Analyze the Category/Topic
            topic_prompt = f"""
            The following is a news article. Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_body}

            ####################

            Task: Classify the article into one of the listed topics.

            Instruction: Try your best to bucket the article into one of these topics. DO NOT write anything that is not listed.
            Key: "topic"
            Value: One of: {topic_list}.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_topic = openai_chat(topic_prompt, model="gpt-3.5-turbo")
            predicted_topic = parsed_topic.get("topic", "")

            # 2b) Analyze the subtopic
            subtopic_prompt = f"""
            The following is a news article on the topic of {predicted_topic}. Read it and perform the task that follows. 
            Respond with a JSON object of key-value pairs.

            ####################

            {article_body}

            ####################

            Task: Classify the article into one of the listed subtopics under the predicted topic.

            Instruction: Try your best to bucket the article into one of these subtopics. Label it as 'Other' if the article does not fit any possible subtopics.
            Key: "subtopic"
            Value: One of {subtopic_dict}.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_subtopic = openai_chat(subtopic_prompt, model="gpt-3.5-turbo")

            # 3) Analyze News Type
            news_type_prompt = f"""
            The following is a news article. Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_body}

            ####################

            Task: Determine the news type of this news article.

            1. Instruction: Classify the above news article into one of three categories: news report, news analysis, or opinion. Each category has distinct characteristics:
            - News Report: Objective reporting on recent events, focusing on verified facts without the writer's personal views.
            - News Analysis: In-depth examination and interpretation of news events, providing context and explaining significance, while maintaining a degree of objectivity.
            - Opinion: Articles reflecting personal views, arguments, or beliefs on current issues, often persuasive and subjective.
            Consider criteria such as language objectivity, focus on facts versus interpretation, author's intent, and article structure. 
            Key: "news_type"
            Value: One of "news report" or "news analysis" or "opinion".


            2. Instruction: Provide a short paragraph to justify your classification, citing specific elements from the text.
            Key: "justification"
            Value: A paragraph of text.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_news_type = openai_chat(news_type_prompt, model="gpt-4o")

            # 4) Analyze Article Tone
            tone_prompt = f"""
            The following is a news article. Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_body}

            ####################

            Task: Determine the overall tone of the article. Is it negative, positive, or neutral?

            1. Instruction: Provide a short paragraph summarizing in what ways the article has a negative or positive tone.
               Key: "reason"
               Value: A paragraph of text.

            2. Instruction: Provide a number from -5 to 5, with -5 indicating a very negative tone and 5 indicating a very positive tone.
               A value of 0 indicates that the article has a neutral tone.
               Key: "tone"
               Value: An integer number from -5 to 5.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_tone = openai_chat(tone_prompt, model="gpt-4o")

            # 4b) Analyze the Title Tone
            title_tone_prompt = f"""
            The following is the title of a news article on the topic of {predicted_topic} ({parsed_subtopic.get("subtopic","")}) in the {parsed_news_type.get("news_type","")} news category. 
            Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_title}

            ####################

            Task: Determine the overall tone of the article title. Is it negative, positive, or neutral?

            1. Instruction: Provide a short paragraph summarizing in what ways the article title has a negative or positive tone.
               Key: "reason"
               Value: A paragraph of text.

            2. Instruction: Provide a number from -5 to 5, with -5 indicating a very negative tone and 5 indicating a very positive tone.
               A value of 0 indicates that the article title has a neutral tone.
               Key: "tone"
               Value: An integer number from -5 to 5.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_title_tone = openai_chat(title_tone_prompt)

            # 5) Analyze the Article Political Lean
            political_lean_prompt = f"""
            The following is a news article. Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_body}

            ####################

            Task: Determine the political leaning of this article within the U.S. political context. 
            Is it supporting the Democrat party or the Republican party? Provide reasoning for your answer.

            1. Instruction: Give a short paragraph summarizing in what ways the article supports the Democrat party or the Republican party.
               Key: "reason"
               Value: A paragraph of text.

            2. Instruction: Give a number from -5 to 5, with -5 indicating strong support for Democrats and 5 indicating strong support for Republicans. 
               A value of 0 indicates that the article has no clear political leaning towards either side.
               Key: "lean"
               Value: An integer number from -5 to 5.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_political_lean = openai_chat(political_lean_prompt)

            # 5b) Analyze the Title Political Lean
            title_political_prompt = f"""
            The following is the title of a news article on the topic of {predicted_topic} ({parsed_subtopic.get("subtopic","")}) in the {parsed_news_type.get("news_type","")} news category.
            Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

            ####################

            {article_title}

            ####################

            Task: Determine the political leaning of this article title within the U.S. political context. 
            Is it supporting the Democrat party or the Republican party? Supporting a party can mean supporting its viewpoints, politicians, or policies. Provide reasoning for your answer.

            1. Instruction: Provide a short paragraph summarizing in what ways the article title supports the Democrat party or the Republican party.
               Key: "reason"
               Value: A paragraph of text.

            2. Instruction: Provide a number from -5 to 5, with -5 indicating strong support for Democrats and 5 indicating strong support for Republicans. 
               A value of 0 indicates that the article title has no clear political leaning towards either side.
               Key: "lean"
               Value: An integer number from -5 to 5.

            Do not return anything except the JSON object of key-value pairs as output.
            """
            parsed_title_political_lean = openai_chat(title_political_prompt)

            # Now combine everything into a single dictionary for this article
            single_article_analysis = {
                "title": article_title,
                "body": article_body,
                # Safe-guarding with .get to avoid KeyError if the JSON is missing keys
                "takeaways": parsed_takeaways.get("takeaways"),
                "topic": predicted_topic,
                "subtopic": parsed_subtopic.get("subtopic"),
                "news_type": parsed_news_type.get("news_type"),
                "news_type_justification": parsed_news_type.get("justification"),
                "article_tone_reason": parsed_tone.get("reason"),
                "article_tone_score": parsed_tone.get("tone"),
                "title_tone_reason": parsed_title_tone.get("reason"),
                "title_tone_score": parsed_title_tone.get("tone"),
                "political_reason": parsed_political_lean.get("reason"),
                "political_lean": parsed_political_lean.get("lean"),
                "title_political_reason": parsed_title_political_lean.get("reason"),
                "title_political_lean": parsed_title_political_lean.get("lean"),
            }

            analysis_results.append(single_article_analysis)

            # print the analysis
            # dispatcher.utter_message(
            #     text=f"Analysis for article: {article_title}\n{single_article_analysis}"
            # )

        # Store `analysis_results` in a slot, e.g., "analysis_results"
        return [SlotSet("analysis_results", analysis_results)]
    

class ActionCotBiasDetection(Action):
    """Refer to bias types, summarize the bias in the article."""
    
    def name(self) -> Text:
        return "action_cot_bias_detection"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Retrieve the news articles from the slot
        news_articles = tracker.get_slot("news_fetched")

        if not news_articles:
            dispatcher.utter_message(text="I don't have any news articles to analyze.")
            return []

        # Ensure articles are in a list format
        articles = news_articles if isinstance(news_articles, list) else [news_articles]

        # Load bias types
        with open("bias_types.json", "r") as f:
            bias_types = json.load(f)

        # Prepare bias types summary for GPT prompt
        bias_summary = "\n".join([
            f"{bias['bias_name']}: {bias['description']}" for bias in bias_types
        ])

        bias_results = []

        for idx, article in enumerate(articles, start=1):
            content = article.get("content", "")
            title = article.get("title", "Untitled")

            # Chat-based prompt structure
            messages = [
                {"role": "system", "content": "You are a helpful assistant trained to detect bias in news articles."},
                {"role": "user", "content": (
                    f"Analyze the following news article for any signs of bias based on these definitions:\n"
                    f"{bias_summary}\n\n"
                    f"Article Title: {title}\n"
                    f"Content: {content}\n\n"
                    f"Please identify and explain any biases detected in the article."
                )}
            ]

            try:
                # Correct OpenAI API call
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  # or "gpt-4"
                    messages=messages,
                    temperature=0.5
                    # max_tokens=500
                )

                analysis = response.choices[0].message["content"].strip()
                result = (
                    f"**Article {idx}:**\n"
                    f"Title: {title}\n"
                    f"Detected Biases: {analysis}\n"
                )
            except Exception as e:
                result = (
                    f"**Article {idx}:**\n"
                    f"Title: {title}\n"
                    f"Error analyzing bias: {e}"
                )

            bias_results.append(result)
            # dispatcher.utter_message(text=result)

        return [SlotSet("cot_bias_detection", bias_results)]


class ActionCotDebiasSummary(Action):
    """Debias multiple summarized news articles using CoT. (Method 3)"""

    def name(self) -> Text:
        return "action_cot_debias_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:

        articles = tracker.get_slot("news_fetched")
        analysis_results = tracker.get_slot("analysis_results")
        bias_detections = tracker.get_slot("cot_bias_detection")

        if not articles:
            dispatcher.utter_message(text="I don't have any summaries to debias.")
            return []

        debiased_summaries = []

        for idx, article in enumerate(articles, start=1):
            title = article.get("title", "")
            content = article.get("content", "")
            analysis = analysis_results[idx - 1] if analysis_results and idx - 1 < len(analysis_results) else {}
            bias_detection = bias_detections[idx - 1] if bias_detections and idx - 1 < len(bias_detections) else ""

            # Prepare system and user prompts
            system_prompt = (
                "You are a professional news editor specialized in unbiased summariation. "
                "Your task is to summarize news articles by eliminating biases while maintaining factual accuracy and clarity."
            )

            user_prompt = (
                f"**Title:** {title}\n\n"
                f"**Content:** {content}\n\n"
                f"**Detected Biases:** {bias_detection}\n\n"
                f"**Analysis:** {analysis.get('body', '')}\n\n"
                "Summarize the article into a paragraph, remove any potential biases or subjective language while preserving the core message."
            )

            try:
                # Call the OpenAI API for debiasing
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                debiased_text = response["choices"][0]["message"]["content"].strip()

                # Provide feedback to the user
                dispatcher.utter_message(
                    text=f"**CoT Debiased Summary {idx}:**\n{debiased_text}"
                )

                # Collect debiased summaries
                debiased_summaries.append({
                    "summary_index": idx,
                    "original_title": title,
                    # "original_content": content,
                    "debiased_summary": debiased_text
                })

            except Exception as e:
                dispatcher.utter_message(
                    text=f"An error occurred while debiasing summary {idx}: {e}"
                )
            
            # use classifier to check if the debiased text is biased
            bias_result = classifier(debiased_text)
            dispatcher.utter_message(
                text=f"**Bias Score of CoT Debiased Summary {idx}:**\n{bias_result}"
            )

        # Store debiased summaries in slot
        return [SlotSet("cot_debias_summary", debiased_summaries)]


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

