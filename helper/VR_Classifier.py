import pickle
import pandas as pd
from os import path

current_dir = path.abspath(path.dirname(__file__))
feature_dict_path = path.join(current_dir, 'feature_dict.pkl')
scaler_model_path = path.join(current_dir, 'std_scaler_2021-05-17 10_52_28.342850.pkl')
model_pipeline_path = path.join(current_dir, 'logistic_regression_model_2021-05-17 10_52_28.342850.pkl')

class VR_Classifier:
    def __init__(self):
        self.classify = pickle.load(open(model_pipeline_path, 'rb'))
        self.encode_dict = pickle.load(open(feature_dict_path, 'rb'))
        self.scaler = pickle.load(open(scaler_model_path, 'rb'))
    
    def encode(self, report):
        report = pd.DataFrame(report)
        for col in report.columns.tolist():
            report[col] = report[col].replace(self.encode_dict[col])
        return report

    def scale(self, report):
        report = self.scaler.transform(report)
        return report

    def predict(self, report):
        label = self.classify.predict(report)
        if label == 1:
            label = 'Higher Risk'
        elif label == 0:
            label = 'Moderate Risk'
        elif label == 2:
            label = 'Lower Risk'
        return label
