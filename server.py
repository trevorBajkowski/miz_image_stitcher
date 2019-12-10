import sys
sys.path.append("..")
from flask import Flask, request, Response, render_template, redirect, send_file
import numpy as np
import cv2
from os import mkdir
from local_utils import ensure_dir
import datetime
from mosaic import final_mosiac

# Initialize the Flask application
app = Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/stitch', methods=['POST'])
def stitch():
    # Get all our form info
    feature_method = request.form['feature']
    matching_method = request.form['matching']
    augs = request.form.getlist('augs')
    scale = float(request.form['scale'])
    files = request.files.getlist('img')

    # Make a unique input and output folder and save the uploaded files
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_directory = app.root_path + '/uploads/' + ts
    out_directory = image_directory + "_out"
    ensure_dir(image_directory)
    mkdir(image_directory)
    for f in files:
        f.save(image_directory + '/' + f.filename)
    
    # Run the mosaic program, which outputs the location of the final png
    output_location = final_mosiac(image_directory, feat_method=feature_method, match_method=matching_method, out_dir=out_directory, edged=False, scale = scale, augmentations=augs)
    return send_file(output_location, mimetype="image/png")

# start flask app
app.run(host="0.0.0.0", port=5000)