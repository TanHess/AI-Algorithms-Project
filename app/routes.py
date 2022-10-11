from flask import render_template, request, flash, redirect, url_for, Blueprint, current_app

view = Blueprint('view',__name__)

@view.route('/',methods=['GET','POST'])
def index():
    return render_template('base.html')