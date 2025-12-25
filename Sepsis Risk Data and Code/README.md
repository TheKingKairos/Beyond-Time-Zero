# Time Zero Heros

Dashboard to streamline the sepsis care workflow at the emergency department of Emory Healthcare Hospitals.

## Features
- Data Cleaning
- EDA
- Patient Ranking: Survival Analysis
- Sepsis Score: Ensemble (XGBoost, MLP)
- Dashboard

## Installation (Mac)
```bash
# clone the repo
git clone #input git link here
cd septic6

# create virtual environment
python3 -m venv sepsis
source sepsis/bin/activate
pip install -r requirements.txt


# disable conda if applicable
# conda deactivate
```

## Data Set Up
First, create a new folder named "data" and, within it, create a subfolder called "MIMIC-ED". Once again, within "MIMIC-ED", create a subfolder named "ed". In that folder, download the MIMIC-IV ED dataset and the labevents.csv from MIMIC-IV/hosp. With this setup, navigate to the data_cleaning folder and read the README.

## Dashboard Frontend and Models
Descriptions and instructions for each part are included in each respective folder. The folder for the frontend design is "folder/Design" and the folder for the models is "algorithm."
