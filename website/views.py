from PIL import Image
import matplotlib
matplotlib.use('Agg')
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

@views.route('/', methods=["GET", "POST"])
def home():
	data = []
	totalCows = {"Date": [], "Total Number of Cows": []}
	if 'username' in session and request.method == 'POST':
		total = findCow(session['herdNumber'])
		cowID = request.form.get('selectCow')
		print(cowID)
		if len(total) != 0:
			for cow in total:
				data.append([cow[0], cow[1], cow[2], cow[4], cow[5]])
			df = pd.DataFrame(data, columns = ['CowID', 'Breed', 'Date of Birth', 'Herd Number', 'Registered Date'])
			#Returning Line Chart of specific cow weight
			data = []
			herdWeights = returnWeightsFromHerd(session['herdNumber'])
			print(herdWeights)
			for weight in herdWeights:
				data.append([weight[3], weight[1], weight[2]])
			weightDF = pd.DataFrame(data, columns=["CowID", "Weight", "Date Weighed"])
			filterPerCowID = weightDF['CowID'] == int(cowID)
			print(weightDF[filterPerCowID])
			plt.plot(weightDF[filterPerCowID]["Date Weighed"], weightDF[filterPerCowID]["Weight"],  linestyle = 'solid')
			plt.xlabel("Weigh Dates", labelpad=14)
			plt.ylabel("Weight", labelpad=14)
			plt.title("Weight of "+str(cowID), y=1.02)

			savePath = os.path.join('website/static', 'SpecificWeight.png')
			plt.savefig(savePath)
			plt.clf()

		return render_template("home.html", cows=weightDF['CowID'].unique(), showWeight=True, showPerBreed=True, showPastYear=True)
	
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

			herdWeights = returnWeightsFromHerd(session['herdNumber'])
			data = []
			print(herdWeights)
			for weight in herdWeights:
				data.append([weight[3], weight[1], weight[2]])
			weightDF = pd.DataFrame(data, columns=["CowID", "Weight", "Date Weighed"])

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

@views.route("/account", methods=["GET","POST"])
def account():
	return render_template("account.html", userID=session['user_id'], herdNo=session['herdNumber'], name=session['name'], email=session['username'])

@views.route("/saveAccount", methods=["GET","POST"])
def saveChangesToAccount():
	if request.method == 'POST':
		userID = request.form.get("user_id")
		herdNumber = request.form.get("herdNumber")
		name = request.form.get("name")
		email = request.form.get("email")
		
		user = findUserForSaving(userID, email)
		herd = findUserWithHerdForSaving(userID, herdNumber)

		if user:
			flash("Email already exists", category='error')
		elif herd:
			flash("Herd Number already exists", category='error')
		else:
			updateUserInDB(herdNumber, name, email, userID)
			flash("Account Changes Saved!", category="success")
			return redirect(url_for('auth.logout'))
	return render_template("account.html", userID=session['user_id'], herdNo=session['herdNumber'], name=session['name'], email=session['username'])
@views.route('/deleteAccount', methods=["GET","POST"])
def deleteAccount():
	if request.method == 'POST':
		herdNo = request.form.get("deleteAccount-HerdNo")
		deleteUserFromDB(herdNo)
		return redirect(url_for('auth.sign_up'))
		
@views.route('/create')
def create():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	return render_template("create.html", cowBreeds=breeds)

@views.route("/medicalProcedures")
def medicalProcedures():
	return render_template("medicalProcedures.html")

@views.route("/saveMedical", methods=["GET", "POST"])
def saveProcedures():
	if request.method == 'POST':
		print(session['cowID'])
		formReturned = []
		data = []
		for key, value in request.form.items():
			formReturned.append(value)
		for i in range(len(formReturned)):
  			if i % 3 == 0:
				  data.append([formReturned[i-2], formReturned[i-1], formReturned[i], session['cowID']])
		createProcedure(data)
	return redirect(url_for('views.weights'))

@views.route("/weights", methods=["GET", "POST"])
def weights():
	return render_template("weights.html")

@views.route("/saveWeights", methods=["GET", "POST"])
def saveWeights():
	if request.method == 'POST':
		formReturned = []
		data = []
		for key, value in request.form.items():
			formReturned.append(value)
		for i in range(len(formReturned)):
  			if i % 2 == 0:
				  data.append([formReturned[i], formReturned[i-1], session['cowID'], session['herdNumber']])
		insertWeight(data)
	flash("Cow Profile Created and Inserted", category='success')
	return redirect(url_for('views.register'))

@views.route('/registerCow', methods=["GET", "POST"])
def register():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
		cow = findCowWithCowID(cowID)
		if cow:
			flash("CowID already exists", category='error')
			return render_template("create.html", valueCowID=cowID, valueBreed=breed, valueDOB=dob,cowBreeds=breeds)
		else:
			session['cowID'] = cowID
			file = request.files['img']
			filename = secure_filename(file.filename)
			savePath = os.path.join('website/uploadImages', filename)
			file.save(savePath)
			model = keras.models.load_model("custom-CNN.h5")
			prediction = model.predict([prepare(savePath)])
			print(prediction)
			if prediction[0][0] == 1.0:
				flash("Cow Picture Detected", category='success')
				createCow(cowID, breed, dob, convert_data(savePath), session["herdNumber"], today.strftime("%Y/%m/%d"))
				os.remove(savePath)
				return render_template("medicalProcedures.html")
			else:
				flash("No Cow Detected, try again.", category='error')
				os.remove(savePath)
				return render_template("create.html", valueCowID=cowID, valueBreed=breed, valueDOB=dob,cowBreeds=breeds)
	return render_template("create.html",cowBreeds=breeds)

@views.route("/scan", methods=["GET", "POST"])
def scan():
	return render_template("scan.html")

@views.route("/scanCow" ,methods=["GET", "POST"])
def searchDatabase():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	cowsInHerd = findCow(herdIn=session["herdNumber"])
	file = request.files['img']
	filename = secure_filename(file.filename)
	savePath = os.path.join('website/uploadImages', filename)
	file.save(savePath)
	transformation = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
	# Resize the images and transform to tensors
	img = Image.open(savePath)
	uploadedImage = transformation(img)
	for cow in cowsInHerd:
		dbImage = transformImage(cow[3])
		load_model = SiameseNetwork().cpu()
		load_optimizer = optim.Adam(load_model.parameters(), lr=0.0006)
		load_checkpoint('Differentmodel.pth',load_model, load_optimizer)
		with torch.no_grad():
			load_model.eval()
			output = load_model(uploadedImage[None, ...], dbImage[None, ...])
			print(output)
			if output.item() < 0.5:
				nparr = np.fromstring(cow[3], np.uint8)
				img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
				cv2.imwrite("website/static/dbReturned.JPG", img_np)
				procedures = returnProcedures(str(cow[0]))
				weights = returnWeights(str(cow[0]))
				os.remove(savePath)
				return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	
	flash("No cows recognised in the DB", category='error')	
	os.remove(savePath)	
	return render_template("scan.html")

@views.route("/deleteCow", methods=["GET", "POST"])
def deleteCow():
	if request.method == 'POST':
		cowID = request.form.get("deleteCow-ID")
		deleteCowFromDB(cowID)
	return render_template("scan.html")

@views.route("/saveEditWeights", methods=["GET","POST"])
def saveEditedWeights():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		numberOfRows = int(len(request.form)/3)
		cowID = request.form.get("editWeight-cowID")
		for i in range(numberOfRows):
			weightID = request.form.get("weightID"+str(i+1))
			weight = request.form.get("weight"+str(i+1))
			date = request.form.get("date"+str(i+1))
			data.append([weightID, weight, date])
		editCowWeights(data)
		cow = findCowWithCowID(cowID)
		procedures = returnProcedures(str(cow[0]))
		weights = returnWeights(str(cow[0]))
		return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

@views.route("/insertNewWeight", methods=["GET","POST"])
def insertNewWeight():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		print(request.form)
		weight = request.form.get("Weight1")
		date = request.form.get("Date1")
		cowID = request.form.get("insertWeight-CowID")
		herdNumber = request.form.get("insertWeight-HerdNumber")
	data.append([weight, date, cowID, herdNumber])
	insertWeight(data)
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)		

@views.route("/deleteWeight", methods=["GET","POST"])
def deleteWeight():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		cowID = request.form.get('deleteWeight-cowID')	
		weightID = request.form.get('weightID')
	deleteCowWeight(weightID)
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

@views.route("saveEditProcedures", methods=["GET","POST"])
def saveEditedProcedures():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		numberOfRows = int(len(request.form)/4)
		cowID = request.form.get("editProcedure-cowID")
		for i in range(numberOfRows):
			procedureID = request.form.get("procedureID"+str(i+1))
			type = request.form.get("type"+str(i+1))
			description = request.form.get("description"+str(i+1))
			date = request.form.get("date"+str(i+1))
			data.append([procedureID, type, description, date])
		editProcedures(data)
		cow = findCowWithCowID(cowID)
		procedures = returnProcedures(str(cow[0]))
		weights = returnWeights(str(cow[0]))
		return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

@views.route("/insertNewProcedure", methods=["GET","POST"])
def insertNewProcedure():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		data = []
		type = request.form.get("Type1")
		desc = request.form.get("Desc1")
		date = request.form.get("Date1")
		cowID = request.form.get("insertProcedure-CowID")
	data.append([desc, date, type, cowID])
	createProcedure(data)
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

@views.route("/deleteProcedure", methods=["GET","POST"])
def deleteProcedure():
	breeds=["Aberdeen Angus", "Belgian Blue", "Limousin", "Simmental", "Friesian", "Hereford", "Charolais", "Shorthorn"]
	if request.method == 'POST':
		cowID = request.form.get('deleteProcedure-cowID')	
		procedureID = request.form.get('deleteProcedure-ID')
	deleteCowProcedure(procedureID)
	cow = findCowWithCowID(cowID)
	procedures = returnProcedures(str(cow[0]))
	weights = returnWeights(str(cow[0]))
	return render_template("scan.html", data=cow, model=True, returnedDBProcedures=procedures, returnedDBWeights=weights, cowBreeds=breeds)	

@views.route('/editCow', methods=["GET", "POST"])
def edit():
	if request.method == 'POST':
		oldCowID = request.form.get('oldCowID')
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
		editCow(oldCowID, cowID, breed, dob)
	return redirect(url_for('views.scan'))

def transformImage(imgArray):
	nparr = np.fromstring(imgArray, np.uint8)
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
	img = Image.fromarray(img_np.astype('uint8'), 'RGB')
	# Resize the images and transform to tensors
	transformation = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])
	img = transformation(img)
	return img

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

def convert_data(file_name):
	with open(file_name, 'rb') as file:
		binary_data = file.read()
	return binary_data