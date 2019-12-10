import sys
sys.path.append("..")
from flask import Flask, request, Response, render_template, redirect, send_file
import jsonpickle
import numpy as np
import cv2
from os import mkdir
import datetime
from mosaic import final_mosiac

# Initialize the Flask application
app = Flask(__name__, template_folder="templates")
app.config["IMAGE_UPLOADS"] = "/uploads"

# route http posts to this method
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/stitch', methods=['POST'])
def stitch():
    files = request.files.getlist('img')
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_directory = app.root_path + '/uploads/' + ts
    mkdir(image_directory)
    for f in files:
        f.save(image_directory + '/' + f.filename)
    
    final_mosiac(image_directory, feat_method='brisk', match_method='flann', out_dir="sample_scene_1_out", edged=False, scale = 0.5)
    image_location = image_directory + "_out/final.png"

    return send_file(image_location, mimetype="image/png")

# start flask app
app.run(host="0.0.0.0", port=5000)