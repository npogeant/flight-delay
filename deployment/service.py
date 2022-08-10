import os
import pickle
from json import load
from typing import Any, Dict

import boto3
import numpy as np
import mlflow
import pandas as pd
import bentoml
import xgboost as xgb
from dotenv import load_dotenv
from bentoml.io import NumpyNdarray
from mlflow.tracking import MlflowClient

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
AWS_S3_BUCKET_ARTIFACTS = os.getenv("AWS_S3_BUCKET_ARTIFACTS")


def load_dv(run_id):
    """Load the DictVectorizer

    Args:
      - run_id : MLflow experiment run id to get the model from the s3 path

    Returns
      The dict vectorizer
    """

    # Select S3 Resource
    s3 = boto3.resource("s3")

    obj = s3.Object(
        AWS_S3_BUCKET_ARTIFACTS.split('/')[2],
        f'1/{run_id}/artifacts/preprocessor/preprocessor.b',
    )
    dv = pickle.load(obj.get()['Body'])

    return dv


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
dv = load_dv(run_id)


def create_bento_service(bento_name, run_id):
    """
    Create a Bento service for the model.

        Args:
            - bento_name : the name of the future bentoml container
            - run_id : mlflow run id

        Returns:
            The model and the bentoml service
    """

    # Load the model
    bentoml.mlflow.import_model(
        bento_name, f'{AWS_S3_BUCKET_ARTIFACTS}{run_id}/artifacts/models'
    )

    model = bentoml.mlflow.get("flight_delay_model:latest").to_runner()

    # Create service with the model
    service = bentoml.Service(name=bento_name + "_service", runners=[model])

    return model, service


bento_name = 'flight_delay_model'
model, service = create_bento_service(bento_name, run_id)


@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(data_: np.ndarray) -> np.ndarray:

    # Preprocess
    X = pd.DataFrame(
        data=data_,
        columns=['DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEP_TIME', 'AIR_TIME'],
    )
    X[['DAY_OF_WEEK', 'DEP_TIME', 'AIR_TIME']] = X[
        ['DAY_OF_WEEK', 'DEP_TIME', 'AIR_TIME']
    ].astype(float)
    data = dv.transform(X.to_dict(orient='records'))

    # Get the prediction
    result = model.predict.run(data)

    return result
