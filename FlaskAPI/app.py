from flask import Flask, render_template, request
import tensorflow as tf
#from keras.models import load_model
#from keras.preprocessing import image


app = Flask(__name__)

dic = {0 : 'Potato___Early_blight', 1 :'Potato___Late_blight' , 2 :'Potato___healthy'}


model = tf.keras.models.load_model(r"C:\Users\Sathwik\Plant Disease classification\weights.23-0.02.hdf5")
model.make_predict_function()

def predict_label(img_path):
	i = tf.keras.preprocessing.image.load_img(img_path, target_size=(256,256))
	i = tf.keras.preprocessing.image.img_to_array(i)
	i = i.reshape(1, 256,256,3)
	p = model.predict_classes(i)
	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"






@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("home.html", prediction = p, img_path = img_path)





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)