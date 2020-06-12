# madgwick_imu
Madgwick gravity estimate in Cython

## Installation
### Prerequisites
- Python >= 3.6

### Virtual environment
Create the virtual environment and install the requirements:

```bash
python3 -m venv env_madgwick
source env_madgwick/bin/activate
pip install -r requirements.txt
```
### Code compilation
Compile the cython code for Madgwick (for now not a clean install, need to rework on the setup):
```bash
python setup.py build_ext --inplace
```

## Run the example
```bash
python example/madgwick_imu_example.py
```
