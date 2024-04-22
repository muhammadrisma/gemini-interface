import os
from flask import Flask, render_template, request

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel

# Create a Flask app instance
app = Flask(__name__)

vertexai.init(project="peppy-linker-420907", location="asia-southeast1")
model = GenerativeModel("gemini-1.0-pro-001")

def generate(user_input):
    responses = model.generate_content(
        contents=user_input,
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 0.9,
            "top_p": 1
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=True,
    )
    return responses 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["prompt"]
        model_responses = generate(user_input)

        response = ""
        for r in model_responses:
            response += r.text

        return render_template("text.html", response=response)
    else:
        return render_template("text.html")

if __name__ == "__main__":
    server_port = os.environ.get("PORT", "8080")
    app.run(debug=False, port=server_port, host="0.0.0.0")