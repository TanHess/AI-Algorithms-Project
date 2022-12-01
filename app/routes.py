from flask import render_template, request, Blueprint, current_app
import numpy as np
from app.helpers import HANDEDNESS, PITCH_TYPES, EVENTS
import tensorflow as tf
from matplotlib.figure import Figure
from io import BytesIO
from base64 import b64encode
import pandas as pd
import os

view = Blueprint('view',__name__)


def get_largest_changes(base_odds, prediction_odds):
    max_increase = (0,prediction_odds[0]-base_odds[0])
    max_decrease = (0,base_odds[0]-prediction_odds[0])
    for i, item in enumerate(base_odds):
        if prediction_odds[i]-item > max_increase[1]:
            max_increase = (i,prediction_odds[i]-item)
        if item-prediction_odds[i] > max_decrease[1]:
            max_decrease = (i,item-prediction_odds[i])
    return max_decrease[0],max_increase[0]



@view.route('/',methods=['GET','POST'])
def index():
    player = "Shohei Ohtani"
    # Get the average stats for the pitcher: 
    csv_path = os.path.join(os.getcwd(), "app", "Models", "savant_ohtani.csv")
    df = pd.read_csv(csv_path)
    df.dropna()
    pitches = [(0,'FF'),(1,'SL'),(2,'FS'),(3,'SI'),(4,'CU'),(5,'FC')]
    counts = [0,0,0,0,0,0]
    for pitch in df['pitch_type']:
        for item in pitches:
            if item[1] == pitch:
                counts[item[0]] += 1
                break
    total = sum(counts)
    PITCH_PERCENTS = [0,0,0,0,0,0]
    pitch_percent_labels = []
    for index, item in enumerate(counts):
        PITCH_PERCENTS[index] = item/total
        pitch_percent_labels.append(str(round((item/total) * 100, 1)) + "%")

    possible_pitches = ["Fastball", "Slider", "Splitter", "Sinker", "Curveball", "Cutter"]
    predict = False
    result = ""
    data = None
    percentages = []
    message = ""
    message2 = ""

    model_path = os.path.join(current_app.root_path, "Models", "Ohtani")
    model = tf.keras.models.load_model(model_path)

    if request.method == "POST":

        # Settup matplotlib values
        fig = Figure(figsize=(15,5))
        ax = fig.subplots(ncols=2)
        
        # Get input data
        count_balls = request.form.get('balls', type=int)
        count_strikes = request.form.get('strikes', type=int)
        handedness = HANDEDNESS.get(request.form.get('handedness'))
        base1 = int(bool(request.form.get('base1')))
        base2 = int(bool(request.form.get('base2')))
        base3 = int(bool(request.form.get('base3')))
        batscore = request.form.get('batscore', type=int)
        pitchscore = request.form.get('pitchscore',type=int)
        outs = request.form.get('outs', type=int)
        prev_desc = EVENTS.get(request.form.get('prev_desc'))

        # Prepare data in a 2d array for predictions
        inputData = [[]]
        inputData[0].append(prev_desc)
        inputData[0].append(handedness)
        inputData[0].append(count_strikes)
        inputData[0].append(count_balls)
        inputData[0].append(base3)
        inputData[0].append(base2)
        inputData[0].append(base1)
        inputData[0].append(outs)
        inputData[0].append(batscore)
        inputData[0].append(pitchscore)

        # Use the ANN model to predict
        inputData = np.array(inputData).astype('float32')
        score = model.predict(inputData)
        result = score.tolist()

        # Get the percentage values of the prediction:
        for item in result[0]:
            percentages.append(str(round(item * 100, 1)) + "%")

        # Plot the data (data[0] is the overall pitched data)
        container = ax[0].bar(possible_pitches, PITCH_PERCENTS)
        ax[0].bar_label(container, labels=pitch_percent_labels)
        ax[0].set_title("Actual Thrown Pitch Distribution")
        container = ax[1].bar(possible_pitches, result[0])
        ax[1].bar_label(container, labels= percentages)
        ax[1].set_title("Prediction Odds")

        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = b64encode(buf.getbuffer()).decode('ascii')

        highest_chance_index = result[0].index(max(result[0]))
        most_likely_pitch = possible_pitches[highest_chance_index]
        message = "The most likely pitch is a " + most_likely_pitch + " with a " + percentages[highest_chance_index] + " chance!"

        max_decrease, max_increase = get_largest_changes(PITCH_PERCENTS, result[0])
        print(max_decrease,max_increase)
        message2 = possible_pitches[max_decrease] + " had the most significant <b>decrease</b> in odds going from "
        message2 += pitch_percent_labels[max_decrease] + " to " + percentages[max_decrease]
        message2 += "<br>" +possible_pitches[max_increase] + " had the most significant <b>increase</b> in odds going from "
        message2 += pitch_percent_labels[max_increase] + " to " + percentages[max_increase]


        # If no errors up until this point, set predict=True to display results:
        predict = True
    
    return render_template('base.html',predict=predict, player=player, message=message, data=data, message2=message2)
