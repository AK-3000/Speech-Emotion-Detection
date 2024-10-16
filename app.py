from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from live_prediction import *

app = Flask(__name__)
## Configure upload location for audio
app.config['UPLOAD_FOLDER'] = "/home/amrutha/Desktop/major project 2/speech_emotion_detection/PROJECT CODE/audio"

## Route for home page
@app.route('/')
def home():
    return render_template('index.html',value="")


@app.route('/results', methods = ['GET', 'POST'])
def results():
	
	if not os.path.isdir("./audio"):		
		os.mkdir("audio")
	if request.method == 'POST':
		try:

			f = request.files['file']
			filename = secure_filename(f.filename)
			filename = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
			f.save(filename)
			#return "successful"
			#f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
		except:
			return render_template('index.html', value="")

	path = '/home/amrutha/Desktop/major project 2/speech_emotion_detection/PROJECT CODE/audio/audio.wav'
	
	df = feature_extraction(path)	
	emotion = predict_emotion_mlp(df)
	return render_template('index.html', value=emotion)
	print(emotion)

if __name__ == "__main__":
	app.run(debug = True)
