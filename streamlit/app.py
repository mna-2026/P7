import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import json

state = st.session_state


def updateExplanationFig():
    """
    Requests the shapley values for the current application reference by
    calling the API, builds the figure showing the main impacts from th shap
    values and stores it in the session state
    """
    if len(state['apiUrl']) == 0:
        return
    if state['curAppId'] is None or state['explanationFig'] is not None:
        return
    url = state['apiUrl'] + '/explanations/' + state['curAppId']
    response = requests.get(url)
    if response.status_code != 200:
        return
    payload = json.loads(response.content)
    fig, ax = plt.subplots()
    shap.plots.bar(shap.Explanation(values=payload['values'],
                                    base_values=payload['base_values'],
                                    data=payload['data'],
                                    feature_names=payload['feature_names']),
                   max_display=11,
                   ax=ax)
    state['explanationFig'] = fig


def showExplanation():
    if state['explanationFig'] is not None:
        dummyCol1, displayCol, dummyCol2 = st.columns([0.7, 3, 1])
        with displayCol:
            st.pyplot(state['explanationFig'], width=800)


def updateScoreAndThreshold():
    """
    Requests the default score and the threshold for the current application
    reference by calling the API and stores the results in the session state
    """
    if len(state['apiUrl']) == 0:
        return
    if state['curAppId'] is None or state['score'] is not None:
        return
    url = state['apiUrl'] + '/scores/' + state['curAppId']
    response = requests.get(url)
    if response.status_code != 200:
        return   
    payload = json.loads(response.content)
    state['threshold'] = payload['threshold']
    state['score'] = payload['score']


def showScore():
    if state['score'] is not None and state['threshold'] is not None:
        colors = ['#2ca02c',
                  '#8ca232',
                  '#dec12c',
                  '#f19623',
                  '#e65518',
                  '#d62728']
        value = state['score'] * 100
        threshold = state['threshold'] * 100
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Default Score (lower is better)'},
            gauge={'axis': {'range': [None, 100],
                            'tickwidth': 2,
                            'tickcolor': '#2A3F5F'},
                   'bar': {'color': '#2A3F5F', 'thickness' : 0.4},
                   'steps': [
                       {'range': [0, threshold - 10],
                        'color': colors[0]},
                       {'range': [threshold - 10, threshold - 10/3],
                        'color': colors[1]},
                       {'range': [threshold - 10/3, threshold],
                        'color': colors[2]},
                       {'range': [threshold, threshold + 10/3],
                        'color': colors[3]},
                       {'range': [threshold + 10/3, threshold + 10],
                        'color': colors[4]},
                       {'range': [threshold + 10, 100],
                        'color': colors[5]}
                   ],
                   'threshold': {'line': {'color': 'red', 'width': 3},
                                 'thickness': 1.0,
                                 'value': threshold}}
        ))
        st.plotly_chart(fig)


def addLogo():
    st.markdown(
        '''
        <style>
            [data-testid="stSidebar"] {
                background-image: url(https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png);
                background-repeat: no-repeat;
                padding-top: 30px;
                background-position: 20px 8px;
                background-size: 228px 90px
            }
        </style>
        ''',
        unsafe_allow_html=True,
    )


def initState():
    if 'appData' not in state:
        state['appData'] = None
    if 'apiUrl' not in state:
        state['apiUrl'] = ''
    if 'appId' not in state:
        state['appId'] = ''
    if 'curAppId' not in state:
        state['curAppId'] = None
    if 'score' not in state:
        state['score'] = None
    if 'threshold' not in state:
        state['threshold'] = None
    if 'explanationFig' not in state:
        state['explanationFig'] = None


def renderHomePage():
    st.header('Information')
    st.markdown('''
        This website assists you in evaluating the application of a customer
        who requests a loan.
        In the backend, a machine learning model calculates a default score
        on the basis of your input.  
        This score is an indication to help you to decide to accept or reject
        the application. The final decision is up to you and may be different
        from what the model suggests.''')
    st.header('Instructions')
    st.markdown('''
        1. Configure the address of the model API in the menu **Settings**''')
    st.markdown('''
        2. In menu **Calculate Credit Score**, input the Application Reference
        of the customer''')
    st.markdown('''
        3. Check the information of the application in tab **Customer
        data**''')
    st.markdown('''
        4. Click on the button **Calculate Score** and check the score in tab
        **Score**. You can additionaly view the explanation of the score in
        tab **Score Interpretation**''')
    # st.write(state)


def renderSettingsPage():
    apiUrl = st.text_input(
        label='Please, input the API URL of the scoring model',
        key='apiUrl',
        type='url',
        persist_state='session'
    )


def updateAppData():
    """
    Updates the current application reference, requests information by calling
    the API and display information.
    """
    if len(state['appId']) > 0:
        if state['appId'] != state['curAppId']:
            state['score'] = None
            state['threshold'] = None
            state['explanationFig'] = None
            state['appData'] = None
            state['curAppId'] = None
            if len(state['apiUrl']) == 0:
                return
            state['curAppId'] = state['appId']
            url = state['apiUrl'] + '/customer-info/' + state['curAppId']
            response = requests.get(url)
            if response.status_code != 200:
                return   
            payload = json.loads(response.content)
            state['appData'] = payload
    else:
        state['score'] = None
        state['threshold'] = None
        state['explanationFig'] = None
        state['appData'] = None
        state['curAppId'] = None


def showAppData():
    if state['appData'] is not None:
        info = state['appData']
        customerInfo = {
            ':color[' + info['CODE_GENDER']['name'] + ']{foreground="black"} / *CODE_GENDER*': info['CODE_GENDER']['value'],
            ':color[' + info['DAYS_BIRTH']['name'] + ']{foreground="black"} / *DAYS_BIRTH*': f'{info['DAYS_BIRTH']['value']} years',
            ':color[' + info['NAME_FAMILY_STATUS']['name'] + ']{foreground="black"} / *NAME_FAMILY_STATUS*': info['NAME_FAMILY_STATUS']['value'],
            ':color[' + info['CNT_CHILDREN']['name'] + ']{foreground="black"} / *CNT_CHILDREN*': info['CNT_CHILDREN']['value'],
            ':color[' + info['NAME_EDUCATION_TYPE']['name'] + ']{foreground="black"} / *NAME_EDUCATION_TYPE*': info['NAME_EDUCATION_TYPE']['value'],
            ':color[' + info['NAME_HOUSING_TYPE']['name'] + ']{foreground="black"} / *NAME_HOUSING_TYPE*': info['NAME_HOUSING_TYPE']['value'],
            ':color[' + info['FLAG_OWN_REALTY']['name'] + ']{foreground="black"} / *FLAG_OWN_REALTY*': info['FLAG_OWN_REALTY']['value'],
        }
        incomeInfo = {
            ':color[' + info['NAME_INCOME_TYPE']['name'] + ']{foreground="black"} / *NAME_INCOME_TYPE*': info['NAME_INCOME_TYPE']['value'],
            ':color[' + info['DAYS_EMPLOYED']['name'] + ']{foreground="black"} / *DAYS_EMPLOYED*': f'{info['DAYS_EMPLOYED']['value']} quarter(s)',
            ':color[' + info['AMT_INCOME_TOTAL']['name'] + ']{foreground="black"} / *AMT_INCOME_TOTAL*': f'{info['AMT_INCOME_TOTAL']['value']:,.2f} $',
        }
        applicationInfo = {
            ':color[' + info['NAME_CONTRACT_TYPE']['name'] + ']{foreground="black"} / *NAME_CONTRACT_TYPE*': info['NAME_CONTRACT_TYPE']['value'],
            ':color[' + info['AMT_CREDIT']['name'] + ']{foreground="black"} / *AMT_CREDIT*': f'{info['AMT_CREDIT']['value']:,.2f} $',
            ':color[' + info['AMT_GOODS_PRICE']['name'] + ']{foreground="black"} / *AMT_GOODS_PRICE*': f'{info['AMT_GOODS_PRICE']['value']:,.2f} $',
            ':color[' + info['AMT_ANNUITY']['name'] + ']{foreground="black"} / *AMT_ANNUITY*': f'{info['AMT_ANNUITY']['value']:,.2f} $',
        }
        custCol, incCol, loanCol = st.columns(3)
        with custCol:
            st.subheader('General Information')
            st.table(customerInfo)
        with incCol:
            st.subheader('Income Information')
            st.table(incomeInfo)
        with loanCol:
            st.subheader('Loan Characteristics')
            st.table(applicationInfo)


def calculateScore():
    updateScoreAndThreshold()
    updateExplanationFig()


def renderScorePage():
    if len(state['apiUrl']) > 0:
        
        with st.container(
            width='stretch',
            height='content',
            horizontal=True,
            horizontal_alignment='right',
            vertical_alignment='bottom'
        ):
            st.text_input(
                label='Please, input the Application Reference',
                key='appId',
                type='default',
                persist_state='session',
                validate=(r'^\d{6}$', 'Application References are 6-digit numbers'),
                on_change=updateAppData
            )
            st.button(
                label='Calculate Score',
                on_click=calculateScore,
            )
        
        dataTab, scoreTab, expTab = st.tabs(['Customer data',
                                             'Score',
                                             'Score Interpretation'])

        with dataTab:
            showAppData()
            
        with scoreTab:
            showScore()

        with expTab:
            showExplanation()
            
    else:
        st.write('Please, configure the settings first')


initState()

st.set_page_config(page_title='Prêt à dépenser',
                   page_icon=':credit_card:',
                   layout='wide')

addLogo()
nav = st.navigation(
    pages=[
        st.Page(renderHomePage,
                title='Home',
                icon=':material/home:'),
        st.Page(renderSettingsPage,
                title='Settings',
                icon=':material/settings:'),
        st.Page(renderScorePage,
                title='Calculate Credit Score',
                icon=':material/credit_score:'),
    ]
)

nav.run()
