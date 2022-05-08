from PIL import Image
import matplotlib
from matplotlib import rcParams, dates
matplotlib.use('Agg')
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from dateutil. relativedelta import relativedelta
from datetime import date, timedelta
from siameseNetwork import SiameseNetwork
from .models import createCow, createProcedure, deleteCowFromDB, deleteCowProcedure, deleteCowWeight, deleteUserFromDB, editCow, editCowWeights, editProcedures, findCow, findCowWithCowID, findUser, findUserForSaving, findUserWithHerd, findUserWithHerdForSaving, insertWeight, returnProcedures, returnWeights, returnWeightsFromHerd, updateUserInDB
import os
import torch.optim as optim
import torchvision.transforms as transforms
from flask import Blueprint, flash, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
import cv2
views = Blueprint('views', __name__)

today = date.today()

#home page route
#showcases graphs based on herd data
@views.route('/', methods=["GET", "POST"])
def home():
	data = []
	totalCows = {"Date": [], "Total Number of Cows": []}
	#making sure user logged
	#making graph for weight of specific cow - selected from dropdown
	if 'username' in session and request.method == 'POST':
		total = findCow(session['herdNumber'])
		cowID = request.form.get('selectCow')
		if len(total) != 0:
			for cow in total:
				data.append([cow[0], cow[1], cow[2], cow[4], cow[5]])
			df = pd.DataFrame(data, columns = ['CowID', 'Breed', 'Date of Birth', 'Herd Number', 'Registered Date'])
			#Returning Line Chart of specific cow weight
			data = []
			herdWeights = returnWeightsFromHerd(session['herdNumber'])
			for weight in herdWeights:
				data.append([weight[3], weight[1], weight[2]])
			weightDF = pd.DataFrame(data, columns=["CowID", "Weight", "Date Weighed"])
			filterPerCowID = weightDF['CowID'] == int(cowID)
			plt.plot(weightDF[filterPerCowID]["Date Weighed"], weightDF[filterPerCowID]["Weight"],  linestyle = 'solid')
			plt.xlabel("Weigh Dates", labelpad=14)
			plt.ylabel("Weight (KG)", labelpad=14)
			plt.title("Weight of "+str(cowID), y=1.02)
			# Format the date into months & days
			plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m')) 
			# Change the tick interval
			plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=3)) 
			# Changes x-axis range
			plt.gca().set_xbound(weightDF[filterPerCowID]["Date Weighed"].iloc[0],weightDF[filterPerCowID]["Date Weighed"].iloc[-1])
			savePath = os.path.join('website/static', 'SpecificWeight.png')
			plt.savefig(savePath)
			plt.clf()

		return render_template("home.html", cows=weightDF['CowID'].unique(), showWeight=True, showPerBreed=True, showPastYear=True)
	#making two graphs when user is logged in
	elif 'username' in session:
		total = findCow(session['herdNumber'])
		if len(total) != 0:
			for cow in total:
					data.append([cow[0], cow[1], cow[2], cow[4], cow[5]])
			df = pd.DataFrame(data, columns = ['CowID', 'Breed', 'Date of Birth', 'Herd Number', 'Registered Date'])
			#Returning Bar Chart based on number of breeds in herd
			df['Breed'].value_counts().plot(kind="bar")
			plt.xlabel("Breed", labelpad=14)
			plt.ylabel("Count of Breed", labelpad=14)
			plt.title("Number of Cows in Herd Per Breed", y=1.02)

			savePath = os.path.join('website/static', 'PerBreed.png')
			plt.savefig(savePath)
			plt.clf()

			#used to display a dropdown of all cows that have weights recorded
			herdWeights = returnWeightsFromHerd(session['herdNumber'])
			data = []
			for weight in herdWeights:
				data.append([weight[3], weight[1], weight[2]])
			weightDF = pd.DataFrame(data, columns=["CowID", "Weight", "Date Weighed"])

			#making graph for the number of cows in herd per month (last year to date)
			for i in range(365, -1, -1):
				retrieveDate = date.today() - timedelta(days=i)
				totalNumber = len(df[df['Registered Date'] <= retrieveDate])
				totalCows['Date'].append(str(retrieveDate.strftime('%d/%m/%y')))
				totalCows["Total Number of Cows"].append(totalNumber)
			totalCowsDF = pd.DataFrame(totalCows)
			totalCowsDF['Date'] = pd.to_datetime(totalCowsDF['Date'])
			monthlyTotalCowsDF = totalCowsDF.resample('M', on='Date').max()
			lastYearDate = date.today() - relativedelta(years=1)
			mask = (monthlyTotalCowsDF['Date'].dt.date > lastYearDate) & (monthlyTotalCowsDF['Date'].dt.date <= date.today())
			pastYearTotalCowsDf = monthlyTotalCowsDF[mask]
			pastYearTotalCowsDf.plot(x="Date", y="Total Number of Cows")
			plt.title('Max Total Number of Cows Per Month (Last 365 Days) ', fontsize=14)
			plt.xlabel('Date', fontsize=14)
			plt.ylabel('Total Number of Cows', fontsize=14)
			plt.grid(True)
			savePath = os.path.join('website/static', 'PastYear.png')
			plt.savefig(savePath)
			plt.clf()
			return render_template("home.html", cows=weightDF['CowID'].unique(), showPerBreed=True, showPastYear=True)

		else:
			return render_template("home.html")
	else:
		return redirect(url_for('auth.login'))

#route to display account details
@views.route("/account", methods=["GET","POST"])
def account():
	return render_template("account.html", userID=session['user_id'], herdNo=session['herdNumber'], name=session['name'], email=session['username'])

#route to save changes made to user account
@views.route("/saveAccount", methods=["GET","POST"])
def saveChangesToAccount():
	if request.method == 'POST':
		#get form data
		userID = request.form.get("user_id")
		herdNumber = request.form.get("herdNumber")
		name = request.form.get("name")
		email = request.form.get("email")
		
		#ensure new herd number or email does not already exist
		user = findUserForSaving(userID, email)
		herd = findUserWithHerdForSaving(userID, herdNumber)

		if user:
			flash("Email already exists", category='error')
		elif herd:
			flash("Herd Number already exists", category='error')
		else:
			#update user record
			updateUserInDB(herdNumber, name, email, userID)
			flash("Account Changes Saved!", category="success")
			#logout user to make them login again with new details
			return redirect(url_for('auth.logout'))
	return render_template("account.html", userID=session['user_id'], herdNo=session['herdNumber'], name=session['name'], email=session['username'])

#route to delete account from DB 
@views.route('/deleteAccount', methods=["GET","POST"])
def deleteAccount():
	if request.method == 'POST':
		herdNo = request.form.get("deleteAccount-HerdNo") #get herd number for account
		deleteUserFromDB(herdNo) #call delete method
		return redirect(url_for('auth.sign_up')) #redirect to sign up page

#route for display "register cow profile page"		
@views.route('/create')
def create():
	#breeds for dropdown
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	return render_template("create.html", cowBreeds=breeds) #render HTML page

#route for display medical procedures page after registering cow
@views.route("/medicalProcedures")
def medicalProcedures():
	return render_template("medicalProcedures.html")

#route for saving medical procedures added during registration of cow
@views.route("/saveMedical", methods=["GET", "POST"])
def saveProcedures():
	if request.method == 'POST':
		formReturned = []
		data = []
		#put all values for every medical procedure into an array
		for key, value in request.form.items():
			formReturned.append(value)
		#procedure has three items - (type, desc and dateCompleted)
		#loop through the array in step of 3 and add details to DB
		for i in range(len(formReturned)):
  			if i % 3 == 0:
				  data.append([formReturned[i-2], formReturned[i-1], formReturned[i], session['cowID']])
		createProcedure(data)
	return redirect(url_for('views.weights')) #redirect to weights page after medical treatments added

#route for display the weights page
@views.route("/weights", methods=["GET", "POST"])
def weights():
	return render_template("weights.html")

#route for saving weights added during registration
@views.route("/saveWeights", methods=["GET", "POST"])
def saveWeights():
	if request.method == 'POST':
		formReturned = []
		data = []
		#put all values for every weight record into an array
		for key, value in request.form.items():
			formReturned.append(value)
		#weight record has weight and date completed
		#loop through array in step of 2 and add details to DB
		for i in range(len(formReturned)):
  			if i % 2 == 0:
				  data.append([formReturned[i], formReturned[i-1], session['cowID'], session['herdNumber']])
		insertWeight(data)
	flash("Cow Profile Created and Inserted", category='success') #display message to tell user cow added
	return redirect(url_for('views.register')) #display register page again

#route for when user creates cow profile
@views.route('/registerCow', methods=["GET", "POST"])
def register():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		#access form data
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
		#ensure cowID is unique
		cow = findCowWithCowID(cowID)
		if cow:
			flash("CowID already exists", category='error')
			return render_template("create.html", valueCowID=cowID, valueBreed=breed, valueDOB=dob,cowBreeds=breeds)
		else:
			session['cowID'] = cowID
			#save uploaded image to "uploadImages" directory
			file = request.files['img']
			filename = secure_filename(file.filename)
			savePath = os.path.join('website/uploadImages', filename)
			file.save(savePath)
			model = keras.models.load_model("custom-CNN.h5")
			#call detection model for image
			prediction = model.predict([prepare(savePath)])
			print(prediction)
			#if cow is detected
			if prediction[0][0] == 1.0:
				flash("Cow Picture Detected", category='success')
				#create cow record
				createCow(cowID, breed, dob, convert_data(savePath), session["herdNumber"], today.strftime("%Y/%m/%d"))
				os.remove(savePath) #remove uploaded image
				return render_template("medicalProcedures.html") #render medical page
			else:
				#no cow detected
				flash("No Cow Detected, try again.", category='error')
				os.remove(savePath) #remove uploaded image
				#display registration page again with inputted values saved to fields
				return render_template("create.html", valueCowID=cowID, valueBreed=breed, valueDOB=dob,cowBreeds=breeds)
	return render_template("create.html",cowBreeds=breeds)

#route for displaying recognition page
@views.route("/scan", methods=["GET", "POST"])
def scan():
	return render_template("scan.html")

#route for when user uploads image for recognizing cow
@views.route("/scanCow" ,methods=["GET", "POST"])
def searchDatabase():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	cowsInHerd = findCow(herdIn=session["herdNumber"]) #get all cows in herd of user logged in
	#save uploaded image
	file = request.files['img']
	filename = secure_filename(file.filename)
	savePath = os.path.join('website/uploadImages', filename)
	file.save(savePath)
	# Resize the images and transform to tensors
	transformation = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
	img = Image.open(savePath)
	uploadedImage = transformation(img)
	#loop through in cow in the herd
	for cow in cowsInHerd:
		#get DB image 
		dbImage = transformImage(cow[3])
		load_model = SiameseNetwork().cpu()
		load_optimizer = optim.Adam(load_model.parameters(), lr=0.0006)
		load_checkpoint('Differentmodel.pth',load_model, load_optimizer)
		with torch.no_grad():
			load_model.eval()
			#compare uploaded image and DB image
			output = load_model(uploadedImage[None, ...], dbImage[None, ...])
			print(output)
			#if same cow
			if output.item() < 0.5:
				nparr = np.fromstring(cow[3], np.uint8)
				img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
				cv2.imwrite("website/static/dbReturned.JPG", img_np)
				#return details for that cow
				procedures = returnProcedures(str(cow[0]))
				weights = returnWeights(str(cow[0]))
				os.remove(savePath) #remove uploaded image
				return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	
	#no cow detected - display scan page again
	flash("No cows recognised in the DB", category='error')	
	os.remove(savePath)	
	return render_template("scan.html")

#route for deleting cow profile
@views.route("/deleteCow", methods=["GET", "POST"])
def deleteCow():
	if request.method == 'POST':
		cowID = request.form.get("deleteCow-ID") #get cowID
		deleteCowFromDB(cowID) #call delete method
	return render_template("scan.html") #display scan page

#route for saving edited weights of cow
@views.route("/saveEditWeights", methods=["GET","POST"])
def saveEditedWeights():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		numberOfRows = int(len(request.form)/3)
		cowID = request.form.get("editWeight-cowID")
		#go through each weight record in table
		for i in range(numberOfRows):
			#gather data and add to array
			weightID = request.form.get("weightID"+str(i+1))
			weight = request.form.get("weight"+str(i+1))
			date = request.form.get("date"+str(i+1))
			data.append([weightID, weight, date])
		editCowWeights(data) #pass array into edit function
		#gather updated cow data from DB and show main modal of cow profile
		cow = findCowWithCowID(cowID)
		procedures = returnProcedures(str(cow[0]))
		weights = returnWeights(str(cow[0]))
		return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

#route for inserting new weight added
@views.route("/insertNewWeight", methods=["GET","POST"])
def insertNewWeight():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		#collect form data
		weight = request.form.get("Weight1")
		date = request.form.get("Date1")
		cowID = request.form.get("insertWeight-CowID")
		herdNumber = request.form.get("insertWeight-HerdNumber")
	data.append([weight, date, cowID, herdNumber])
	insertWeight(data) #call function to insert to DB
	#gather updated cow data from DB and show main modal of cow profile
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)		

#route for deleting specific record of weight in DB
@views.route("/deleteWeight", methods=["GET","POST"])
def deleteWeight():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		#get weight details
		cowID = request.form.get('deleteWeight-cowID')	
		weightID = request.form.get('weightID')
	deleteCowWeight(weightID) #call delete function
	#gather updated cow data from DB and show main modal of cow profile
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

#route to save procedures edited by user
@views.route("saveEditProcedures", methods=["GET","POST"])
def saveEditedProcedures():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		numberOfRows = int(len(request.form)/4)
		cowID = request.form.get("editProcedure-cowID")
		#go through each record in the table
		for i in range(numberOfRows):
			#gather data and store to array
			procedureID = request.form.get("procedureID"+str(i+1))
			type = request.form.get("type"+str(i+1))
			description = request.form.get("description"+str(i+1))
			date = request.form.get("date"+str(i+1))
			data.append([procedureID, type, description, date])
		#call function to save edited records
		editProcedures(data)
		#gather updated cow data from DB and show main modal of cow profile
		cow = findCowWithCowID(cowID)
		procedures = returnProcedures(str(cow[0]))
		weights = returnWeights(str(cow[0]))
		return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

#route to save new medical record to DB
@views.route("/insertNewProcedure", methods=["GET","POST"])
def insertNewProcedure():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		#gather form data
		type = request.form.get("Type1")
		desc = request.form.get("Desc1")
		date = request.form.get("Date1")
		cowID = request.form.get("insertProcedure-CowID")
	data.append([desc, date, type, cowID]) #add to array
	createProcedure(data) #call insert method
	#gather updated cow data from DB and show main modal of cow profile
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

@views.route("/deleteProcedure", methods=["GET","POST"])
def deleteProcedure():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		#individual medical record details
		cowID = request.form.get('deleteProcedure-cowID')	
		procedureID = request.form.get('deleteProcedure-ID')
	deleteCowProcedure(procedureID) #call delete procedure function
	#gather updated cow data from DB and show main modal of cow profile
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

#route for updating details on specific cow
@views.route('/editCow', methods=["GET", "POST"])
def edit():
	if request.method == 'POST':
		#collect form data
		oldCowID = request.form.get('oldCowID')
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
		editCow(oldCowID, cowID, breed, dob) #call edit function
	return redirect(url_for('views.scan'))

def transformImage(imgArray):
	nparr = np.fromstring(imgArray, np.uint8)
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
	img = Image.fromarray(img_np.astype('uint8'), 'RGB')
	# Resize the images and transform to tensors
	transformation = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])
	img = transformation(img)
	return img

#function to load pytorch siamese model
def load_checkpoint(path,model, optimizer):
	save_path = path
	state_dict = torch.load(save_path, map_location=torch.device('cpu'))
	model.load_state_dict(state_dict['model_state_dict'])
	optimizer.load_state_dict(state_dict['optimizer_state_dict'])
	val_loss = state_dict['val_loss']
	print(f'Model loaded from <== {save_path}')
	
	return val_loss

def prepare(img):
	img_array = cv2.imread(img)  # read in the image, convert to grayscale
	new_array = cv2.resize(img_array, (180, 180))  # resize image to match model's expected sizing
	return new_array.reshape(-1, 180, 180, 3)

#converting image to binary to save in DB
def convert_data(file_name):
	with open(file_name, 'rb') as file:
		binary_data = file.read()
	return binary_data