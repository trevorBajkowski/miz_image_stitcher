import sys
sys.path.append("..")
from flask import Flask, request, Response, render_template, redirect, send_file
import numpy as np
import cv2
from os import mkdir
import datetime
from mosaic import final_mosiac

# Initialize the Flask application
app = Flask(__name__, template_folder="templates")

# route http posts to this method
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/stitch', methods=['POST'])
def stitch():
    params = request.form
    feature_method = params['feature']
    matching_method = params['matching']
    augs = params.getlist('augs')
    files = request.files.getlist('img')
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_directory = app.root_path + '/uploads/' + ts
    out_directory = image_directory + "_out"
    mkdir(image_directory)
    for f in files:
        f.save(image_directory + '/' + f.filename)
    
    output_location = final_mosiac(image_directory, feat_method=feature_method, match_method=matching_method, out_dir=out_directory, edged=False, scale = 0.5, augmentations=augs)
    image_location = image_directory + "_out/final.png"

    return send_file(output_location, mimetype="image/png")

# start flask app
app.run(host="0.0.0.0", port=5000)