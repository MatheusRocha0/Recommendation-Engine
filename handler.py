import os
from flask import Flask, jsonify, request, jsonify
import pickle
import pandas as pd
import surprise

engine = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def predict():
   test_json = request.get_json()

   if test_json:
      df_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
      preds = []
      col1 = df_raw.columns[0]
      col2 = df_raw.columns[1]
      for i in range(len(df_raw)):
         pred = engine.predict(df_raw[col1][i], df_raw[col2][i]).est
         preds.append(pred)

      df_raw["user_rating"] = preds
      return df_raw.to_json(orient = "records")

   else:
      return jsonify({"Greetings": "Hello, World"})

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port)
  
   
