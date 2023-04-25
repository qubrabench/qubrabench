cd ./test

echo "Running tests for KIT"
python3 -m pytest -m kit

echo "Running tests for RUB"
python3 -m pytest -m rub