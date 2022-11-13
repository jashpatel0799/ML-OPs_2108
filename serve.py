from flask import Flask
from flask import request, jsonify
from PIL import Image
from numpy import asarray
from joblib import load
from skimage.transform import resize

app = Flask(__name__)
svm_clf = load("svm_gamma_0.0005_C_0.5.joblib")

@app.route("/", methods = ["GET"])
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/sum", methods = ["POST"])
def sum():
    print(request.json)
    x = request.json["x"]
    y = request.json["y"]
    z = x + y
    return jsonify({'sum':z})


@app.route("/predict", methods = ['POST'])
def predict():
    # image = Image.open("9_img.jpg")
    # print(image)

    # n_img = asarray(image)
    # print(n_img)


    n_img = request.json['image']

    

    # rs = resize(n_img[0], (8,8))

    # print(rs.shape)

    # sc = svm_clf.predict(rs.flatten().reshape(1,-1))
    sc = svm_clf.predict([n_img])

    print(sc)

    return str(sc[0])


# app.run(host=0.0.0.0, port = 6000)
# curl http://127.0.0.1:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}'