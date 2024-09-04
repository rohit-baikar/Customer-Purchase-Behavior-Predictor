from flask import *
import pickle
import joblib
import numpy as np

app=Flask(__name__)

model = pickle.load(open("cpb1.pkl", "rb"))
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=["GET", "POST"])
def home():
	if request.method == "POST":
		age=int(request.form["age"])
		sex = int(request.form["sex"])
		ai = int(request.form["ai"])
		purchases = int(request.form["purchases"])
		pc = int(request.form["pc"])
		tsow = float(request.form["tsow"])
		lpm = int(request.form["lpm"])
		da = int(request.form["da"])
		print(age,sex,ai,purchases,pc,tsow,lpm,da)
		d = np.array([[age,sex,ai,purchases,pc,tsow,lpm,da]])
		nd = scaler.transform(d)
		ans = model.predict(nd)
		ans = "Purchase" if ans[0] == 1 else "No Purchase"
		return render_template("home.html", msg=ans)
		
	else:
		return render_template("home.html")

if __name__=='__main__':
    app.run(debug=True, use_reloader=True)             