# Flight Delay Predictor Pipeline

[//]: <> (# prefect deployment build ./training_flow.py:main_flow -n flight-delay-training -i process -t ml -t training # prefect deployment apply ./flight_training-deployment.yaml)
[//]: <> (# prefect deployment build ./monitoring_flow.py:monitoring_flow -n flight-delay-monitoring -i process -t ml -t monitoring # prefect deployment apply ./flight_monitoring-deployment.yaml)