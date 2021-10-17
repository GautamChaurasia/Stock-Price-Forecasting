from flask import Flask,render_template,request,jsonify,Response
from ARIMA import Predict as pd

app =Flask(__name__)
#x_plot = [dt.datetime.now() + dt.timedelta(days=i) for i in range(30)]

global decode, sym
decode = {'ACN':'Accenture', 'AMZN':'Amazon Inc.', 'F':'Ford Motors', 'RELIANCE.NS':'Reliance', 'TCS.NS': 'Tata Consultancy Services'}
sym = {'₹':('RELIANCE.NS', 'TCS.NS'), '$':('ACN', 'AMZN', 'F')}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/<stock_name>')
def disp(stock_name):
    obj = pd(stock = stock_name, st='2016-01-01')
    ts_data = obj.get_data()
    cp_fig = obj.visualize_data(ts_data)

    if stock_name in sym['₹']:
        curr = '₹'
    else: curr = '$'

    prev_datelist = list(ts_data[::-1].index)
    for i in range(len(prev_datelist)):
        prev_datelist[i] = str(prev_datelist[i])[:10]

    return render_template('stock.html', stock_name=decode[stock_name], stock = stock_name, prev_cp = ts_data[::-1].round(1), prev_datelist = prev_datelist, cp_fig = cp_fig, curr = curr)

@app.route('/<stock_name>_fc', methods=["GET","POST"])
def predict(stock_name):
    obj = pd(stock = stock_name, st='2016-01-01')
    ts_data = obj.get_data()
    cp_fig = obj.visualize_data(ts_data)
    obj.stationarize_data(ts_data)
    fc, datelist = obj.forecast(period='1m')
    path = obj.save_fig()

    prev_datelist = list(ts_data[::-1].index)
    for i in range(len(prev_datelist)):
        prev_datelist[i] = str(prev_datelist[i])[:10]

    for i in range(len(datelist)):
        datelist[i] = str(datelist[i])[:10]

    if stock_name in sym['₹']:
        curr = '₹'
    else: curr = '$'

    return render_template('forecast.html', stock_name = decode[stock_name], len = len(fc), datelist = datelist, prev_datelist = prev_datelist, prev_cp = ts_data[::-1].round(1), cp_fig = cp_fig, fc_fig = path, fc_vals = fc.round(1), curr = curr)


if __name__ == "__main__":
    app.run(debug=True)
