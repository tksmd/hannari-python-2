# Hannari Python #2 Demonstration

This project includes demonstration code for the presentation at [Hannari Python #2](https://hannari-python.connpass.com/event/74633/) at Jan 19th 2018.

# Setup

## Create venv

Requires Python 3.6 or later. 

    pyenv local 3.6.3

then create venv environment like this

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.lock

## Start Jupyter

Create jupyter kernel for this project as follows

    python -m ipykernel install --user --name hannari-python-2 --display-name "Hannari Python #2"
    jupyter notebook
