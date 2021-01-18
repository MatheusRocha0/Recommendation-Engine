from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods = ["POST", "GET"])
def home():
   try:
      to_predict = request.form.to_dict()
      to_predict = list(to_predict.values())
      to_predict = list(map(int, to_predict))
      engine = pickle.load(open("model.pkl", "rb"))
      prediction = engine.predict(to_predict[0], to_predict[1]).est
  
   except:
      prediction = "Prediction"

   return render_template("index.html", prediction = prediction)

if __name__ == "__main__":
   app.run()