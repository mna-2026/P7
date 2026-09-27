import pickle
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()
# Load selected model
with open(r'model_step7.pkl', 'rb') as f:
    model = pickle.load(f)
# Load the computed threshold
with open(r'model_threshold_step7.pkl', 'rb') as f:
    threshold = pickle.load(f)
# Load the dataset: it is the database of the applications
with open(r'dataset_test.pkl', 'rb') as f:
    data = pickle.load(f)
# Load the shap values
with open(r'shap_values_test.pkl', 'rb') as f:
    shapValues = pickle.load(f)

@app.get('/health-check')
async def healthCheck():
    return 'The API is up and running'

@app.get('/scores/{appId}')
async def score(appId: int):
    if appId not in data.index:
        errMsg = str(appId)
        errMsg += ' is not a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    response = {
        'threshold': threshold,
        'score': model.predict_proba(data.loc[[appId]])[0][1].item()
    }
    return response

@app.get('/explanations/{appId}')
async def shapExplanation(appId: int):
    if appId not in data.index:
        errMsg = str(appId)
        errMsg += ' is not a valid Application Reference.'
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
