import datetime
from flask import Flask, request, jsonify
from helper.VR_Classifier import VR_Classifier
from helper.encoder import JSONEncoder
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config["MONGO_URI"] = "mongodb://localhost:27017/report_db"
mongodb_client = PyMongo(app)
db = mongodb_client.db

# Check DB
@app.route('/')
def home():
    reports = db.todos.find()
    return jsonify([JSONEncoder().encode(report) for report in reports])

# VRA from Report Route
@app.route('/send-report', methods=['POST'])
def report():
    datelog = str(datetime.datetime.now())
    nik = request.args['nik']
    vt = request.args['violence_type']
    rel = request.args['relation']
    vic = request.args['victim_age']
    ag = request.args['agressor_age']
    par = request.args['previous_abuse_report']
    lt = request.args['living_together']
    sc = request.args['short_chronology']
    report = [(rel, vic, ag, par, lt)]

    classifier = VR_Classifier()
    encoded_report = classifier.encode(report)
    scaled_report = classifier.scale(encoded_report)
    risk_level = classifier.predict(scaled_report)

    db.todos.insert_one({
        'date_log': datelog,
        'nik' : nik,
        'violence_type': vt,
        'relation': rel,
        'victim_age': vic,
        'agressor_age': ag,
        'prev_abuse_report': par,
        'living_together': lt,
        'short_chronology': sc,
        'risk_level': risk_level
    })

    return jsonify({
        'Report Date' : datelog,
        'NIK' : nik,
        'Violence Type': vt,
        'Relation': rel,
        'Victim Age': vic,
        'Agressor Age': ag,
        'Previous Abuse Report': par,
        'Living Together': lt,
        'Short Chronology': sc,
        'Risk Level': risk_level
    })

if __name__ == '__main__':
    app.run()