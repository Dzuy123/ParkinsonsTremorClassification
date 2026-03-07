# Parkinson Project

Machine learning in medicine project for Parkinson hand-motion exercise classification.

## Setup

### Clone this repo to your machine

```bash
git clone https://github.com/Dzuy123/ParkinsonsTremorClassification.git
cd ParkinsonsTremorClassification
```

Then run one of the commands below.

### Fedora

```bash
sudo dnf install -y python3 python3-devel gcc gcc-c++ make git
python3 -m venv ParkinsonVenv
source ParkinsonVenv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.lock.txt
python -m ipykernel install --user --name=ParkinsonVenv --display-name "Python (ParkinsonVenv)"
```

### Ubuntu / Vast.ai

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential git
python3 -m venv ParkinsonVenv
source ParkinsonVenv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.lock.txt
python -m ipykernel install --user --name=ParkinsonVenv --display-name "Python (ParkinsonVenv)"
```

### Windows PowerShell

```powershell
py -m venv ParkinsonVenv
.\ParkinsonVenv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.lock.txt
python -m ipykernel install --user --name=ParkinsonVenv --display-name "Python (ParkinsonVenv)"
```

If error, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run

### Fedora / Ubuntu / Vast.ai

```bash
source ParkinsonVenv/bin/activate
jupyter lab
```

### Windows PowerShell

```powershell
.\ParkinsonVenv\Scripts\Activate.ps1
jupyter lab
```

## Verify installation

```bash
python -c "import numpy, pandas, sklearn, scipy, matplotlib, seaborn, tqdm; print('OK')"
```


## File structure
celosia@fedora:~/Tom/VSCode/ML_Med/Parkins
onProject$ tree -a -L 2 
.
├── data
│   ├── data
│   ├── .~lock.test.csv#
│   ├── sample_submit.csv
│   ├── test.csv
│   └── train.csv
├── .git
├── .gitignore
├── parkinson_eda.ipynb
├── ParkinsonVenv
├── README.md
├── requirements.in
└── requirements.lock.txt

