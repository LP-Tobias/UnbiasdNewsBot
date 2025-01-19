# Unbiased News Bot

A Rasa-based chatbot designed to fetch news, detect and mitigate biases in articles, and provide neutral summaries. Our goal is to demonstrate **debiased** news generation by integrating an LLM and specialized techniques such as **Chain of Thought (CoT)** prompting.

## Getting Started

1. **Install Dependencies**  
Please also read the instructions in the requirements.
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
      rasa run --enable-api --cors "*" --debug
     ```
   - In another terminal, launch the action server:
     ```bash
     rasa run actions
     ```
   - In the last terminal, launch Flask:
     ```bash
     python app.py
     ```



## Troubleshooting

- **API Key Issues**: Ensure that your API keys are correctly set in the `.env` file and that they have the necessary permissions.

