import pandas as pd

training_flow = __import__(
    'experiments and training.training_flow', fromlist=('preprocessing')
)


def test_preprocessing():
    # Actual Dataframe (preprocessed)
    actual_df = training_flow.preprocessing(['data/flight_status_01_2022.csv'])
    # Expected Dataframe
    data = ([(None), (None), (None), (None), (None), (None)],)
    columns = [
        'DAY_OF_WEEK',
        'OP_CARRIER',
        'ORIGIN',
        'DEP_TIME',
        'AIR_TIME',
        'IS_DELAY',
    ]
    expected_result = pd.DataFrame(data, columns=columns)

    assert actual_df.shape[1] == expected_result.shape[1]


def test_splitting():
    # pylint: disable=unused-variable
    # Actual Result
    training_flow.spliting.fn(
        ['data/flight_status_01_2022.csv'], ['data/flight_status_02_2022.csv']
    )
    X_train, X_val, y_train, y_val, dv = training_flow.spliting.fn(
        ['data/flight_status_01_2022.csv'], ['data/flight_status_02_2022.csv']
    )
    actual_result = (X_train.nnz, X_val.nnz)
    # Expected Result
    expected_result = (2517645, 2365730)

    assert actual_result == expected_result
