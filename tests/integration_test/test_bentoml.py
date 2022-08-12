import ast

import requests
from deepdiff import DeepDiff

API_ENDPOINT = "http://127.0.0.1:3000/predict"
inputs = '[[7, "YX", "PHL", 1400, 56.0]]'

# Send the image to the API
response = requests.post(API_ENDPOINT, data=inputs)

actual_response = float(ast.literal_eval(response.text)[0])
expected_response = 0.6070127487182617

print(f'Actual Response : {actual_response}')
print(f'Expected Response : {expected_response}')

diff = DeepDiff(actual_response, expected_response, significant_digits=1)

assert 'type_changes' not in diff
