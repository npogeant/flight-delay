cd deployment

ECHO 'Waiting for the model service to be up...'

bentoml serve service.py:service &
sleep 20

cd ..
cd tests/integration_test/

python test_bentoml.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    kill $(ps -A | grep "bentoml" | awk '{print $1}')
    exit ${ERROR_CODE}
fi

kill $(ps -A | grep "bentoml" | awk '{print $1}')
