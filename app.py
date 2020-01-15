from keras.models import model_from_json
from flask import Flask, flash, render_template, redirect, request, send_file, url_for
import cv2
import numpy as np

app = Flask(__name__)
loaded_model = None


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if image_file:
            passed = False
            try:
                filename = image_file.filename
                filepath = '/home/neo/Desktop/Flask_Image_Classification/data/'+filename
                image_file.save(filepath)
                passed = True
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
	preds = ''
	try:
		
		img = cv2.imread('/home/neo/Desktop/Flask_Image_Classification/data/'+filename)
		img = cv2.resize(img,(64,64))
		img = np.array(img)
		img = img.reshape(1,64,64,3)
		pred = loaded_model.predict_classes(img)

		if(pred == 1):
		    preds = 'DOG'
		    print('=============================')
		    print('\t\tDOG')
		    print('=============================')
			
		else:
			preds = 'CAT'
			print('=============================')
			print('\t\tCAT')
			print('=============================')
	except Exception as e:
		print(str(e))		
		
	return render_template('result.html',text=preds)

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html')

if __name__ == "__main__":
	json_file = open('/home/neo/Desktop/Flask_Image_Classification/Static/weights/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model._make_predict_function()
	# load weights into new model
	loaded_model.load_weights("/home/neo/Desktop/Flask_Image_Classification/Static/weights/model.h5")
	print("Loaded model from disk")
	app.run('0.0.0.0',debug=True)