import pickle
import numpy as np
from zipfile import ZipFile
import shap
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()
# Load selected model
with open(r'../data/model_step7.pkl', 'rb') as f:
    model = pickle.load(f)
# Load the computed threshold
with open(r'../data/model_threshold_step7.pkl', 'rb') as f:
    threshold = pickle.load(f)
# Load the dataset: it is the database of the applications
with ZipFile(r'../data/dataset_test.pkl.zip') as z:
    with z.open(r'dataset_test.pkl') as f:
        data = pickle.load(f)
with ZipFile(r'../data/human_friendly_dataset_test.pkl.zip') as z:
    with z.open(r'human_friendly_dataset_test.pkl') as f:
        humanFriendlyData = pickle.load(f)
# Initialize the SHAP explainer
explainer = shap.TreeExplainer(model,
                               feature_perturbation='tree_path_dependent')
# Compute the SHAP values                             
shapValues = explainer(data)
print('API loaded')


@app.get('/health-check')
async def healthCheck():
    return 'The API is up and running'


@app.get('/customer-info/{appId}')
async def customerInfo(appId: int):
    if appId not in data.index or appId not in humanFriendlyData.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    info = humanFriendlyData.loc[appId]
    los = info['DAYS_EMPLOYED']
    response = {
        'CODE_GENDER': {
            'name': 'Sex',
            'value': info['CODE_GENDER']
        },
        'DAYS_BIRTH': {
            'name': 'Age',
            'value': info['DAYS_BIRTH'].item()
        },
        'NAME_FAMILY_STATUS': {
            'name': 'Family Status',
            'value': info['NAME_FAMILY_STATUS']
        },
        'CNT_CHILDREN': {
            'name': 'Number of Children',
            'value': info['CNT_CHILDREN'].item()
        },
        'NAME_EDUCATION_TYPE': {
            'name': 'Education',
            'value': info['NAME_EDUCATION_TYPE']
        },
        'NAME_HOUSING_TYPE': {
            'name': 'Housing',
            'value': info['NAME_HOUSING_TYPE']
        },
        'FLAG_OWN_REALTY': {
            'name': 'Real Estate Owner',
            'value': info['FLAG_OWN_REALTY']
        },
        'NAME_INCOME_TYPE': {
            'name': 'Income Type',
            'value': info['NAME_INCOME_TYPE']
        },
        'DAYS_EMPLOYED': {
            'name': 'Length of Service',
            'value': 'Unknown' if np.isnan(los) else int(los)
        },
        'AMT_INCOME_TOTAL': {
            'name': 'Total Income',
            'value': round(info['AMT_INCOME_TOTAL'], 2).item()
        },
        'NAME_CONTRACT_TYPE': {
            'name': 'Cash or Revolving',
            'value': info['NAME_CONTRACT_TYPE']
        },
        'AMT_CREDIT': {
            'name': 'Credit Amount',
            'value': round(info['AMT_CREDIT'], 2).item()
        },
        'AMT_GOODS_PRICE': {
            'name': 'Goods Price (Consumer Loan)',
            'value': round(info['AMT_GOODS_PRICE'], 2).item()
        },
        'AMT_ANNUITY': {
            'name': 'Annuity',
            'value': round(info['AMT_ANNUITY'], 2).item()
        }
    }
    return response


@app.get('/scores/{appId}')
async def score(appId: int):
    if appId not in data.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    response = {
        'threshold': threshold,
        'score': model.predict_proba(data.loc[[appId]])[0][1].item()
    }
    return response


@app.get('/explanations/{appId}')
async def shapExplanation(appId: int):
    if appId not in data.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    idx = data.index.get_loc(appId)
    sv = shapValues[idx]
    return {
        'values': sv.values.tolist(),
        'base_values': sv.base_values.tolist(),
        'data': sv.data.tolist(),
        'feature_names': sv.feature_names
    }


if __name__ == "__main__":
    print('Starting server')
    uvicorn.run(app, host='0.0.0.0', port=8000)
