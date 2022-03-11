from PIL import Image
import numpy as np
import torch

from siameseNetwork import SiameseNetwork
from .models import createCow, editCow, findCow
import os
import torch.optim as optim
import torchvision.transforms as transforms
from flask import Blueprint, flash, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
import cv2
views = Blueprint('views', __name__)

@views.route('/')
def home():
	if 'username' in session:
		return render_template("home.html")
	else:
		return redirect(url_for('auth.login'))

@views.route('/create')
def create():
	return render_template("create.html")

@views.route('/registerCow', methods=["GET", "POST"])
def register():
	if request.method == 'POST':
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
		file = request.files['img']
		filename = secure_filename(file.filename)
		savePath = os.path.join('website/uploadImages', filename)
		file.save(savePath)
		model = keras.models.load_model("custom-CNN.h5")
		prediction = model.predict([prepare(savePath)])
		print(prediction)
		if prediction[0][0] == 1.0:
			flash("Cow Profile Created", category='success')
			createCow(cowID, breed, dob, convert_data(savePath), session["herdNumber"])
			os.remove(savePath)
			return redirect(url_for('views.register'))
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
	print(len(cowsInHerd))
	for cow in cowsInHerd:
		file = request.files['img']
		filename = secure_filename(file.filename)
		savePath = os.path.join('website/uploadImages', filename)
		file.save(savePath)
		transformation = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
		# Resize the images and transform to tensors
		img = Image.open(savePath)
		#numpydata = np.asarray(img)
		#img = Image.fromarray(numpydata.astype('uint8'), 'RGB')
		uploadedImage = transformation(img)
		dbImage = transformImage(cow[3])
		load_model = SiameseNetwork().cuda()
		load_optimizer = optim.Adam(load_model.parameters(), lr=0.0006)
		load_checkpoint('Differentmodel.pth',load_model, load_optimizer)
		with torch.no_grad():
			load_model.eval()
			output = load_model(uploadedImage[None, ...].cuda(), dbImage[None, ...].cuda())
			print(output)
			os.remove(savePath)
			if output.item() > 0.5:
				pred = 1
				return render_template("scan.html")
			else:
				pred = 0
				nparr = np.fromstring(cow[3], np.uint8)
				img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
				cv2.imwrite("website/static/dbReturned.JPG", img_np)
				return render_template("scan.html", data=cow, model=True)
	return ""

@views.route('/editCow', methods=["GET", "POST"])
def edit():
	if request.method == 'POST':
		oldCowID = request.form.get('oldCowID')
		cowID = request.form.get('cowID')
		breed = request.form.get('breed')
		dob = request.form.get("dob")
		herd = request.form.get("herdNumber")
		editCow(oldCowID, cowID, breed, dob, herd)
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