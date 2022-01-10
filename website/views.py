from flask import Blueprint, render_template, session, redirect, url_for

views = Blueprint('views', __name__)

@views.route('/')
def home():
    if 'username' in session:
        return render_template("home.html")
    else:
        return redirect(url_for('auth.login'))