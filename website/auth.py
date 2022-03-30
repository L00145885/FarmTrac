from flask import Blueprint, render_template, request, flash, redirect, session, url_for
from .models import createUser, findUser, findUserWithHerd
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = findUser(email)
        if user:
            if check_password_hash(user[-1], password):
                session['username'] = user[0]
                session['herdNumber'] = user[1]
                flash("Logged In Successfully", category='success')
                return redirect(url_for('views.home'))
            else:
                flash("Incorrect password, try again.", category='error')
        else:
            flash("Email does not exist", category='error')
    return render_template("login.html")

@auth.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('auth.login'))

@auth.route('/sign-up', methods=["GET", "POST"])
def sign_up():
    if request.method == "POST":
        herdNumber = request.form.get('herdNumber')
        name = request.form.get('name')
        email = request.form.get('email')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        
        user = findUser(email)
        herd = findUserWithHerd(herdNumber)
        if user:
            flash("Email already exists", category='error')
        elif herd:
            flash("Herd Number already exists", category='error')
        elif len(herdNumber) < 2:
            flash("Herd Number is too short.", category="error")
        elif len(email) < 6:
            flash("Email must be greater than 5 characters.", category="error")
        elif len(name) < 2:
            flash("Full name must be greater than 1 character.", category="error")
        elif password1 != password2:
            flash("Passwords do not match.", category="error")
        elif len(password1) < 7:
            flash("Passwords must be at least 7 characters.", category="error")
        else:
            createUser(herdNumber,name,email,passIn=generate_password_hash(password1, method="sha256"))
            flash("Account Created!", category="success")
            return redirect('/')
    return render_template("sign_up.html")