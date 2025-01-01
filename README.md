# Unbiased News Bot

A Rasa-based chatbot designed to fetch news, detect and mitigate biases in articles, and provide neutral summaries. Our goal is to demonstrate **debiased** news generation by integrating an LLM and specialized techniques such as **Chain of Thought (CoT)** prompting.

## Project Structure

```
.
├── actions
│   └── actions.py        # Custom action logic (fetch news, debiasing, CoT, etc.)
├── config.yml            # Rasa NLU pipeline and policies
├── credentials.yml       # Credentials for external APIs
├── domain.yml            # Rasa domain (intents, slots, responses, etc.)
├── endpoints.yml         # Action server endpoints
├── data
│   ├── nlu.yml           # NLU training data
│   ├── stories.yml       # Story-based training data
│   └── rules.yml         # Rule-based dialogue data
└── requirements.txt      # Python dependencies
```

## Current Progress

**Log — 1/1/2025 (Peiheng)**  
- Implemented basic stories and actions for core functionality (fetching news, summarizing, etc.).  

## To-Do List

1. **Expand NLU & Stories**  
   - Add more training examples in `nlu.yml` and more stories in `stories.yml` so the bot can handle a wider range of user inputs.  

2. **Improve Bias Detection**  
   - Implement a more sophisticated approach to identify and handle biased terms in the news (possibly via LLM prompts or specialized libraries).  

3. **Enhance Chain of Thought (CoT)**  
   - Refine the action logic to better illustrate the step-by-step reasoning in debiasing.  

4. **Showcase Debiasing Effect**  
   - Provide clearer before-and-after examples to demonstrate how the bot reduces bias in its summaries.  

5. **Integrate a Flask App**  
   - Wrap the current Rasa chatbot inside a Flask application for web deployment or further expansions (e.g., custom UI, additional endpoints).

## Getting Started

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Up Environment**  
   - Add your API keys (NewsAPI, OpenAI, etc.) to a `.env` file.
3. **Train the Model**  
   ```bash
   rasa train
   ```
4. **Run the Bot**  
   - In one terminal, start the Rasa server:
     ```bash
     rasa shell
     ```
   - In another terminal, launch the action server:
     ```bash
     rasa run actions
     ```
