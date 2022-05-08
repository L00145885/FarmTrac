from flask import Blueprint, render_template, request, flash, redirect, session, url_for
from .models import createUser, findUser, findUserWithHerd
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

#route for login page
@auth.route('/login', methods=["GET", "POST"])
def login():
    #if user submits login credentials
    if request.method == 'POST':
        email = request.form.get('email') #access email input
        password = request.form.get('password') #access password input

        user = findUser(str(email)) #check if any user in DB has email
        #if user returned
        if user:
            #compare password hash in DB and hash of password entered
            if check_password_hash(user[-1], password):
                #store user details in session
                session['user_id'] = user[0]
                session['username'] = user[3]
                session['herdNumber'] = user[1]
                session['name'] = user[2]
                flash("Logged In Successfully", category='success')
                return redirect(url_for('views.home')) #redirect to home screen
            else:
                flash("Incorrect password, try again.", category='error') #flash message to enter another password
        else:
            flash("Email does not exist", category='error') #flash message to enter another email
    #renter login page when accessing GET request
    return render_template("login.html")

#route for when user logs out
@auth.route('/logout')
def logout():
    session.clear() #clear session data
    return redirect(url_for('auth.login')) #display login screen

#route for sign up
@auth.route('/sign-up', methods=["GET", "POST"])
def sign_up():
    herdNumber = ''
    name = ''
    email = ''
    password1 = ''
    password2 = ''
    #if POST request
    if request.method == "POST":
        #getting all form data
        herdNumber = request.form.get('herdNumber')
        name = request.form.get('name')
        email = request.form.get('email')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        
        #returning user with herd number or email - need to be unique
        user = findUser(email)
        herd = findUserWithHerd(str(herdNumber))
        #validation handling
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
            #all data validated
            #create user record in DB without hashed password
            createUser(herdNumber,name,email,passIn=generate_password_hash(password1, method="sha256"))
            flash("Account Created!", category="success")
            return redirect('/') #redirect to home page
    return render_template("sign_up.html", valueHerdNo=herdNumber, valueName=name, valueEmail=email, valuePass1=password1, valuePass2=password2)