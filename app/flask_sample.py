
# -*- coding: utf-8 -*-

from flask import Flask,render_template,request, make_response, jsonify,request, redirect, url_for, send_from_directory, session
import os
import werkzeug.utils
from werkzeug import secure_filename
from datetime import datetime
import alphabet_cnn_original
from keras import backend as back
import MainPredict as MP

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
UPLOAD_DIR = 'your/directory/path'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif','jpeg'])
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

    
@app.route('/converter')
def hello():
    return render_template('convertapp_original.html')


# result of predict
@app.route('/send', methods=['POST'])
def send():

    result = None
    back.clear_session()
    
    if request.method != 'POST':
        return '''error'''
    x=request.query_string
    y=x.decode("utf-8").split('&')
    print(y)
    img_file = request.files['img_file']
    if img_file != True and allowed_file(img_file.filename) != True:
        return ''' NOT ALLOWED EXTENSION '''
        
    filename = secure_filename(img_file.filename)
    img_file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
    img_path = UPLOAD_DIR + '/'+ filename
            
    # predict
    cnn = MP.MainPredict()
    result = cnn.predict(img_path)               
    
    if result == None:
        return ''' FAILED TO RECIGNIZE TEXT... '''

    return str(result)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=80)
    