from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from datetime import date, timedelta
from siameseNetwork import SiameseNetwork
from .models import createCow, createProcedure, editCow, findCow, insertWeight
import os
import torch.optim as optim
import torchvision.transforms as transforms
from flask import Blueprint, flash, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
import cv2
views = Blueprint('views', __name__)

today = date.today()

@views.route('/')
def home():
	if 'username' in session:
		return render_template("home.html")
	else:
		return redirect(url_for('auth.login'))

@views.route('/create')
def create():
	return render_template("create.html")

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
				  data.append([formReturned[i], formReturned[i-1], session['cowID']])
		insertWeight(data)
	flash("Cow Profile Created and Inserted", category='success')
	return redirect(url_for('views.register'))

@views.route('/registerCow', methods=["GET", "POST"])
def register():
	if request.method == 'POST':
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
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
			return render_template("create.html", valueCowID=cowID, valueBreed=breed, valueDOB=dob)
	return render_template("create.html")

@views.route("/scan", methods=["GET", "POST"])
def scan():
	return render_template("scan.html")

@views.route("/scanCow" ,methods=["GET", "POST"])
def searchDatabase():
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
			os.remove(savePath)
			if output.item() < 0.5:
				nparr = np.fromstring(cow[3], np.uint8)
				img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
				cv2.imwrite("website/static/dbReturned.JPG", img_np)
				return render_template("scan.html", data=cow, model=True)			
	return render_template("scan.html")

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
	state_dict = torch.load(save_path)
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