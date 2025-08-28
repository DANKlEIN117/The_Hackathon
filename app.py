from flask import Flask, render_template, request
from ai_engine.model import query_symptoms


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    symptoms = request.form.get("symptoms")
    if not symptoms:
        return render_template("result.html", result="Please enter your symptoms.")
    
    result = query_symptoms(symptoms)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
