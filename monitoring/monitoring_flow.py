import os
import pickle
from datetime import datetime

import boto3
import numpy as np
import mlflow
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from mlflow.tracking import MlflowClient
from evidently.dashboard import Dashboard
from prefect.task_runners import SequentialTaskRunner
from evidently.dashboard.tabs import (
    DataDriftTab,
    CatTargetDriftTab,
    ClassificationPerformanceTab,
)
from evidently.pipeline.column_mapping import ColumnMapping

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_S3_BUCKET_ARTIFACTS = os.getenv("AWS_S3_BUCKET_ARTIFACTS")


def get_files_list(month, year, AWS_S3_BUCKET):
    """Get s3 bucket path"""

    month = int(month)
    year = int(year)
    return f'{AWS_S3_BUCKET}flight_status_{month:02d}_{year:04d}.csv'


@task
def get_data():
    """Get data from S3 and preprocess it"""

    # Reference data is the dataset before the last dataset
    reference_data_dates = [
        datetime.now().month - 3
        if datetime.now().month < 10
        else datetime.now().month - 3,
        datetime.now().year if datetime.now().month > 3 else datetime.now().year - 1,
    ]

    # Current data is the last dataset
    current_data_dates = [
        datetime.now().month - 4
        if datetime.now().month < 10
        else datetime.now().month - 4,
        datetime.now().year if datetime.now().month > 4 else datetime.now().year - 1,
    ]

    reference_data_path_S3 = get_files_list(
        reference_data_dates[0], reference_data_dates[1], AWS_S3_BUCKET
    )
    current_data_path_S3 = get_files_list(
        current_data_dates[0], current_data_dates[1], AWS_S3_BUCKET
    )

    reference_data = pd.read_csv(reference_data_path_S3)
    current_data = pd.read_csv(current_data_path_S3)

    reference_data.dropna(
        axis=0, how='any', inplace=True
    )  # remove rows that do not have arriving time and not cancelled
    reference_data['IS_DELAY'] = np.where(reference_data['DEP_DELAY'] <= 0, 0, 1)
    reference_data = reference_data[
        ['DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEP_TIME', 'AIR_TIME', 'IS_DELAY']
    ]

    current_data.dropna(
        axis=0, how='any', inplace=True
    )  # remove rows that do not have arriving time and not cancelled
    current_data['IS_DELAY'] = np.where(current_data['DEP_DELAY'] <= 0, 0, 1)
    current_data = current_data[
        ['DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEP_TIME', 'AIR_TIME', 'IS_DELAY']
    ]

    return reference_data, current_data


@task
def load_model_and_dv(run_id):
    """Load the model and DictVectorizer

    Args:
      - run_id : MLflow experiment run id to get the model from the s3 path

    Returns
      The model and dict vectorizer
    """

    # Select S3 Resource
    s3 = boto3.resource("s3")

    obj = s3.Object(
        AWS_S3_BUCKET_ARTIFACTS.split('/')[2],
        f'1/{run_id}/artifacts/preprocessor/preprocessor.b',
    )
    dv = pickle.load(obj.get()['Body'])

    model = mlflow.pyfunc.load_model(
        f'{AWS_S3_BUCKET_ARTIFACTS}{run_id}/artifacts/models'
    )

    return model, dv


@task
def get_predictions(dataframe, model, dv):
    """Get predictions from the model

    Args:
      - dataframe : a pandas dataframe
      - model
      - dv : dictvectorizer

    Returns:
      A modified dataframe
    """

    input_data = dv.transform(
        dataframe[
            ['DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEP_TIME', 'AIR_TIME']
        ].to_dict(orient='records')
    )

    y_preds = model.predict(input_data)
    dataframe['preds'] = [0 if pred <= 0.5 else 1 for pred in y_preds]
    dataframe.rename(columns={'IS_DELAY': 'target'}, inplace=True)

    return dataframe


@task
def build_report(reference_data, current_data):
    """Build a dashboard with reports inside from Evidently

    Args:
      - reference_data : pandas dataframe with referenced data
      - current_data : pandas dataframe with current data

    Returns:
      An Evidently report
    """

    mapping = ColumnMapping(
        prediction="preds",
        target="target",
        numerical_features=['DAY_OF_WEEK', 'DEP_TIME', 'AIR_TIME'],
        categorical_features=['OP_CARRIER', 'ORIGIN'],
        datetime_features=[],
    )

    report = Dashboard(
        tabs=[DataDriftTab(), CatTargetDriftTab(), ClassificationPerformanceTab()]
    )
    report.calculate(
        reference_data=reference_data, current_data=current_data, column_mapping=mapping
    )

    return report


@task
def save_html_report(result, result_path):
    """Save a report as an html"""

    result.save(result_path)


@task
def save_report(result_path):
    """Save a report in an s3 bucket"""

    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=result_path, Bucket=AWS_S3_BUCKET.split('/')[2], Key=result_path
    )


@flow(task_runner=SequentialTaskRunner())
def monitoring_flow():
    reference_data, current_data = get_data()

    client = MlflowClient(MLFLOW_TRACKING_URI)
    registered_model = client.search_registered_models(
        filter_string="name='flight-delay-classifier'"
    )
    run_id = [
        model.run_id
        for model in registered_model[0].latest_versions
        if model.current_stage == 'Production'
    ][
        0
    ]  # '0383e5c8ab394626bec4810f8a48fe36'
    model, dv = load_model_and_dv(run_id)

    reference_data = get_predictions(reference_data, model, dv)
    current_data = get_predictions(current_data, model, dv)

    report = build_report(reference_data, current_data)

    result_path = 'reports/report.html'
    save_html_report(report, result_path)
    save_report(result_path)


if __name__ == '__main__':
    monitoring_flow()
