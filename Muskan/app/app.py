# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kaleido

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound func() function.
def func():

    path = "static/answers.svg"
    model = load('model.joblib')
    make_picture('AgesAndHeights.pkl',model, floats_string_to_np_arr('3, 5, 15'), path)
  
    return render_template('index.html')



def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data['Age']
  data = data[ages > 0]
  ages = data['Age']
  heights = data['Height']
  x_new = np.array(list(range(19))).reshape(19, 1)
  preds = model.predict(x_new)

  fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)','y': 'Height (inches)'})

  fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

  new_preds = model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='yellow', size=20, line=dict(color='purple', width=2))))
  
  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()

def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)

if __name__ == "__main__":
    app.run

