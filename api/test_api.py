from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_init():
    assert str(type(api.model)) == '<class \'xgboost.sklearn.XGBClassifier\'>'
    assert str(type(api.threshold)) == '<class \'numpy.float64\'>'
    assert api.data.shape == (48744, 425)
    assert api.humanFriendlyData.shape == (48744, 14)
    assert api.shapValues.shape == (48744, 425)


def test_healthcheck():
    response = client.get('/health-check')
    assert response.status_code == 200
    assert response.json() == 'The API is up and running'


def test_score_ok():
    response = client.get('/scores/100001')
    expectedJson = {
        'threshold': 0.4984014290384948,
        'score': 0.31223005056381226
    }
    assert response.status_code == 200
    assert response.json() == expectedJson


def test_score_ko():
    response = client.get('/scores/100000')
    expectedJson = {
        'detail': '100000 not found. It doesn\'t seem a valid Application Reference.'
    }
    assert response.status_code == 404
    assert response.json() == expectedJson


def test_customerinfo_ok():
    response = client.get('/customer-info/100001')
    expectedJson = {
        'CODE_GENDER': {
            'name': 'Sex',
            'value': 'F'
        },
        'DAYS_BIRTH': {
            'name': 'Age',
            'value': 52
        },
        'NAME_FAMILY_STATUS': {
            'name': 'Family Status',
            'value': 'Married'
        },
        'CNT_CHILDREN': {
            'name': 'Number of Children',
            'value': 0
        },
        'NAME_EDUCATION_TYPE': {
            'name': 'Education',
            'value': 'Higher education'
        },
        'NAME_HOUSING_TYPE': {
            'name': 'Housing',
            'value': 'House / apartment'
        },
        'FLAG_OWN_REALTY': {
            'name': 'Real Estate Owner',
            'value': 'Y'
        },
        'NAME_INCOME_TYPE': {
            'name': 'Income Type',
            'value': 'Working'
        },
        'DAYS_EMPLOYED': {
            'name': 'Length of Service',
            'value': 25
        },
        'AMT_INCOME_TOTAL': {
            'name': 'Total Income',
            'value': 135000
        },
        'NAME_CONTRACT_TYPE': {
            'name': 'Cash or Revolving',
            'value': 'Cash loans'
        },
        'AMT_CREDIT': {
            'name': 'Credit Amount',
            'value': 568800
        },
        'AMT_GOODS_PRICE': {
            'name': 'Goods Price (Consumer Loan)',
            'value': 450000
        },
        'AMT_ANNUITY': {
            'name': 'Annuity',
            'value': 20560.5
        }    
    }
    assert response.status_code == 200
    assert response.json() == expectedJson
   

def test_customerinfo_ko():
    response = client.get('/customer-info/100000')
    expectedJson = {
        'detail': '100000 not found. It doesn\'t seem a valid Application Reference.'
    }
    assert response.status_code == 404
    assert response.json() == expectedJson


def test_explanations_ok():
    response = client.get('/explanations/100001')
    assert response.status_code == 200
    json = response.json()
    assert 'values' in json.keys() and len(json['values']) == 425
    assert 'base_values' in json.keys() and json['base_values'] == 0.09060005843639374
    assert 'data' in json.keys() and len(json['data']) == 425
    assert 'feature_names' in json.keys() and len(json['feature_names']) == 425


def test_explanations_ko():
    response = client.get('/explanations/100000')
    expectedJson = {
        'detail': '100000 not found. It doesn\'t seem a valid Application Reference.'
    }
    assert response.status_code == 404
    assert response.json() == expectedJson