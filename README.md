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

## Usage

- **Request News**: Ask the bot to fetch news on a specific topic.
  ```plaintext
  User: I want news about technology.
  Bot: Here are the latest news articles about technology...
  ```

- **Detect Bias**: The bot will analyze the fetched news for biases and provide a summary.
  ```plaintext
  User: Summarize the news.
  Bot: The summary of the news articles is...
  ```

## Troubleshooting

- **Contradicting Rules or Stories**: If you encounter errors related to contradicting rules or stories, ensure that your `rules.yml` and `stories.yml` files are consistent and do not conflict with each other.

- **API Key Issues**: Ensure that your API keys are correctly set in the `.env` file and that they have the necessary permissions.

