echo "Creating virtual environment with Python 3.11..."
rm -r .venv # Remove existing virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
echo "Installing requirements.txt..."
pip install --target=.venv/lib/python3.11/site-packages -r requirements.txt
echo "Virtual environment is now created. Run 'source .venv/bin/activate' to activate it, then `func start` to start the function server locally."