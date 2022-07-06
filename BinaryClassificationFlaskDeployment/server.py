from flask import Flask, render_template, request
from flask.views import MethodView
from predict_input import PredictOutcome

app = Flask(import_name= __name__)

class HomePage(MethodView):
    def get(self):
        return render_template("home.html")

    def post(self):
        input_value = request.form.get('input')
        str_to_list = input_value.split(',')
        int_list = []
        for ele in str_to_list:
            int_list.append(float(ele))
        print(int_list)
        po = PredictOutcome(input_value= [ int_list ]) # as predict expects 2D array
        output = po.predict()
        return render_template('home.html', result= True, output_text= output)

app.add_url_rule(rule= '/', view_func= HomePage.as_view(name= "home_page"))

app.run(debug= True)