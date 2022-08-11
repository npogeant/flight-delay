import os
import pickle
from datetime import datetime

import numpy as np
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from prefect import flow, task
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from hyperopt.pyll import scope
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer

load_dotenv()


def get_files_list(month, year, AWS_S3_BUCKET):
    """Get s3 bucket path"""

    month = int(month)
    year = int(year)
    return f'{AWS_S3_BUCKET}flight_status_{month:02d}_{year:04d}.csv'


def preprocessing(
    files: list,
):
    '''Preprocess the data, drop missing values, encode target

    Args:
    - files : csv path list

    Returns:
     A preprocessed dataframe
    '''

    if len(files) > 1:
        files_dict = {}
        for num, file in enumerate(files):
            files_dict[num] = pd.read_csv(file)

        dataframe = pd.concat(list(files_dict.values()))
    else:
        dataframe = pd.read_csv(files[0])

    dataframe.dropna(
        axis=0, how='any', inplace=True
    )  # remove rows that do not have arriving time and not cancelled

    # Create Categorical Delay Column
    dataframe['IS_DELAY'] = np.where(dataframe['DEP_DELAY'] <= 0, 0, 1)

    dataframe = dataframe[
        ['DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEP_TIME', 'AIR_TIME', 'IS_DELAY']
    ]

    return dataframe


@task
def spliting(file_for_training: list, file_for_validation: list):
    '''Split the data in training and validation arrays

    Args:
    - file_for_training : training csv path list
    - file_for_validation : validation csv path list

    Returns:
     X_train, X_val, y_train, y_val, dv
    '''

    df_train = preprocessing(file_for_training)
    df_val = preprocessing(file_for_validation)

    categorical = ['OP_CARRIER', 'ORIGIN']
    numerical = ['DEP_TIME', 'DAY_OF_WEEK', 'AIR_TIME']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = 'IS_DELAY'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv


@task
def hyperparameter_tuning(train, valid, y_val):  # pylint: disable=useless-return
    '''Tune the model with HyperOpt and log runs in mlflow

    Args
      - train : xgb DMatrix of training features
      - valid : xgb DMatrix of validation features
      - y_val : array of validation target values

    Returns
    '''

    def objective(params):
        with mlflow.start_run():

            mlflow.set_tag("model", "XGBClassifier")
            mlflow.set_tag("run type", "hyperparameter tuning")

            mlflow.log_params(params)

            model = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50,
            )

            y_pred = model.predict(valid)

            precision = precision_score(y_val, y_pred.round())

            mlflow.log_metric("precision", precision_score(y_val, y_pred.round()))
            mlflow.log_metric("recall", recall_score(y_val, y_pred.round()))
            mlflow.log_metric("roc auc", roc_auc_score(y_val, y_pred.round()))

        return {'loss': -precision, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'binary:logistic',
        'seed': 42,
    }

    best_result = fmin(  # pylint: disable=unused-variable
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=1, trials=Trials()
    )
    return


@task
def find_and_train_best_model(train, valid, y_val, dv, client):
    '''Select the best model from the previous runs and train the model
    to keep it in the S3 bucket artifacts folder.

    Args
      - train : xgb DMatrix of training features
      - valid : xgb DMatrix of validation features
      - y_val : array of validation target values
      - dv : fitted DictVectorizer
      - client : MlflowClient

    Returns
    '''

    with mlflow.start_run():

        filter_query = "tags.`run type` = 'hyperparameter tuning'"
        hypertuning_runs = client.search_runs(
            experiment_ids=["1"],
            filter_string=filter_query,
            order_by=["metrics.precision DESC", "attributes.start_time DESC"],
            max_results=50,
        )
        best_run_params = hypertuning_runs[0].data.params

        mlflow.set_tag("model", "XGBClassifier")
        mlflow.set_tag("run type", "best model")

        mlflow.log_params(best_run_params)

        model = xgb.train(
            params=best_run_params,
            dtrain=train,
            num_boost_round=1,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50,
        )

        y_pred = model.predict(valid)

        mlflow.log_metric("precision", precision_score(y_val, y_pred.round()))
        mlflow.log_metric("recall", recall_score(y_val, y_pred.round()))
        mlflow.log_metric("roc auc", roc_auc_score(y_val, y_pred.round()))

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(model, artifact_path="models")


@task
def register_and_set_stage_model(client):
    '''Register the model trained and set its stage
    If the model is better than the production one,
    put it in staging, if not put it in the archives.

    Args

    Returns
    '''

    best_model_runs = client.search_runs(
        experiment_ids=["1"],
        filter_string="tags.`run type` = 'best model'",
        order_by=["attributes.start_time DESC"],
        max_results=5,
    )
    run_id = best_model_runs[0].info.run_id  # save the most recent best model run id

    # Register the model
    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/models", name='flight-delay-classifier'
    )
    version = mv.version

    # Get the production model's run id
    registered_model = client.search_registered_models(
        filter_string="name='flight-delay-classifier'"
    )
    production_run_id = [
        model.run_id
        for model in registered_model[0].latest_versions
        if model.current_stage == 'Production'
    ]
    if len(production_run_id) > 0:
        # Get precision metric of both run
        production_run = client.get_run(production_run_id[0])
        production_run_precision = production_run.data.metrics['precision']

        best_run = client.get_run(run_id)
        best_run_precision = best_run.data.metrics['precision']

        if best_run_precision > production_run_precision:
            model_version = version
            newstage = "Production"
            client.transition_model_version_stage(
                name='flight-delay-classifier',
                version=model_version,
                stage=newstage,
                archive_existing_versions=False,
            )
            print(
                f'The version {version} of flight-delay-classifier model has been set to {newstage}'
            )
        else:
            model_version = version
            newstage = "Archived"
            client.transition_model_version_stage(
                name='flight-delay-classifier',
                version=model_version,
                stage=newstage,
                archive_existing_versions=False,
            )
            print(
                f'The version {version} of flight-delay-classifier model has been set to {newstage}'
            )
    else:
        model_version = version
        newstage = "Production"
        client.transition_model_version_stage(
            name='flight-delay-classifier',
            version=model_version,
            stage=newstage,
            archive_existing_versions=False,
        )
        print(
            f'The version {version} of flight-delay-classifier model has been set to {newstage}'
        )


@flow(task_runner=SequentialTaskRunner())
def main_flow(
    training_data_month: list = [
        (
            datetime.now().month - 3
            if datetime.now().month < 10
            else datetime.now().month - 3,
            datetime.now().year
            if datetime.now().month > 3
            else datetime.now().year - 1,
        )
    ],
    validation_data_month: list = [
        (
            datetime.now().month - 4
            if datetime.now().month < 10
            else datetime.now().month - 4,
            datetime.now().year
            if datetime.now().month > 4
            else datetime.now().year - 1,
        )
    ],
):
    # pylint: disable=too-many-locals
    # pylint: disable=dangerous-default-value
    """Run the main flow"""
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
    MLFLOW_TRACKING_USERNAME = os.getenv(  # pylint: disable=unused-variable
        'MLFLOW_TRACKING_USERNAME'
    )
    MLFLOW_TRACKING_PASSWORD = os.getenv(  # pylint: disable=unused-variable
        'MLFLOW_TRACKING_PASSWORD'
    )
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")  # pylint: disable=unused-variable

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(MLFLOW_TRACKING_URI)

    training_data = []
    for month, year in training_data_month:
        training_data.append(get_files_list(month, year, AWS_S3_BUCKET))

    validation_data = []
    for month, year in validation_data_month:
        validation_data.append(get_files_list(month, year, AWS_S3_BUCKET))

    X_train, X_val, y_train, y_val, dv = spliting(
        file_for_training=training_data, file_for_validation=validation_data
    )
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    hyperparameter_tuning(train, valid, y_val)
    find_and_train_best_model(train, valid, y_val, dv, client)
    register_and_set_stage_model(client)


if __name__ == '__main__':
    main_flow()
