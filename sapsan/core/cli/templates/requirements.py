TEMPLATE = """
catalyst>=21.5
Click>=7.1.2
graphviz>=0.14
h5py>=2.10.0
jupyter>=1.0.0
jupytext>=1.11
matplotlib>=3.3.2
mlflow>=1.20.1
notebook>=6.4.3
numpy>=v1.19.2
opencv-python>=4.5.1
pandas>=1.1.0
Pillow>=8.1.0
plotly>=5.2.0
pytest>=6.2
safitty>=1.3
scikit-image>=0.17.2
scikit-learn>=0.23.2
scipy>=1.5.2
six>=1.15.0
streamlit==0.84.2
tornado>=6.1.0
"""


def get_requirements_template():
    return TEMPLATE
