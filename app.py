from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__, template_folder='templates')
CORS(app)

# Rasa server URL
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    sender_id = request.json.get("sender", "default")

    # Forward user message to Rasa
    response = requests.post(RASA_URL, json={"sender": sender_id, "message": user_message})
    rasa_response = response.json()

    # Extract bot replies
    bot_messages = [msg.get("text") for msg in rasa_response if "text" in msg]

    return jsonify({"messages": bot_messages})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
