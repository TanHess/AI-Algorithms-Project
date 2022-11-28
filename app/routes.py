from flask import render_template, request, flash, redirect, url_for, Blueprint, current_app
import tensorflow as tf
import numpy as np

    
    
with open("Models/Baseball_ohtani.py") as file:
    print(file.read())
    
model = tf.saved_model.load('Desktop/School/Fall2022/ArtificialIntelligence/SemesterProject/AI-Algorithms-Project/app/Model/Ohtani')

view = Blueprint('view',__name__)

@view.route('/',methods=['GET','POST'])
def index():
    predict = False
    prediction = ""
    if request.method == "POST":
        # Past 5 pitches (to be added to past_pitches)
        p1 = (request.form.get("pitchtype1"), request.form.get("pitchzone1", type=int))
        p2 = (request.form.get("pitchtype2"), request.form.get("pitchzone2", type=int))
        p3 = (request.form.get("pitchtype3"), request.form.get("pitchzone3", type=int))
        p4 = (request.form.get("pitchtype4"), request.form.get("pitchzone4", type=int))
        p5 = (request.form.get("pitchtype5"), request.form.get("pitchzone5", type=int))
        past_pitches = [p1, p2, p3, p4, p5]

        count_balls = request.form.get('balls', type=int)
        count_strikes = request.form.get('strikes', type=int)
        handedness = request.form.get('handedness')
        base1 = bool(request.form.get('base1'))
        base2 = bool(request.form.get('base2'))
        base3 = bool(request.form.get('base3'))
        score = request.form.get('score', type=int)
        print("past Pitches", past_pitches)
        print("Count balls:", count_balls)
        print("Count strikes:", count_strikes)
        print("Handedness:", handedness)
        print("First base:", base1)
        print("Second base:", base2)
        print("Third base:", base3)
        print("Score:",score)

        # If no errors up until this point, set predict=True to display results:
        predict = True
        pitches_str = ""
        for item in past_pitches:
            pitches_str += str(item[0])
            pitches_str += str(item[1])
            pitches_str += "\n"
        prediction = pitches_str + str(count_balls) + "\n" + str(count_strikes) + "\n"  + str(handedness) + "\n"  + str(base1) + "\n"  + str(base2) + "\n"  + str(base3) + "\n"  + str(score)
    
    return render_template('base.html',predict=predict, prediction=prediction)
