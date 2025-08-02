from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def check_data_drift(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html("drift_report.html")
