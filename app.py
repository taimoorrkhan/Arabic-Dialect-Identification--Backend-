
#Falsk Libs
from flask import Flask,request, jsonify, render_template
import traceback
import subprocess
import traceback 
import time
import os

#Loading the audio file class for prediction
from klaam import SpeechClassification
model = SpeechClassification()


app=Flask(__name__)

def predict_wav_file(wav_file_path):
    predicted_label = model.classify(wav_file_path,True)
    return predicted_label


@app.route('/')
def home():
    return "The ADI API is up and running!"

@app.route('/predict',methods=['POST',"GET"])
def predict():
    
    file = request.files['']

    wav_filename = "testing_wav_file.wav"

    if file.filename.split(".")[1] == 'mp4':

        try:
            print("Saving.")
            filename = "user_file.mp4"
            file.save(filename)
            print("Saved.")
            subprocess.check_output(
                'ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(filename, "testing_wav_file"), stderr=subprocess.STDOUT,shell=True)
            print("Converted.")
        except:
            print(traceback.print_exc())

    else:
        file.save(wav_filename)
    
    response = predict_wav_file(wav_filename)

    try:
        os.remove("testing_wav_file.wav")
        os.remove("user_file.mp4")
    except:
        pass
    
    return response

if __name__=="__main__":
    app.run(host ='0.0.0.0' ,debug=False)



