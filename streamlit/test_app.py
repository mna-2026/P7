from streamlit.testing.v1 import AppTest
import app

appTest = AppTest.from_file('app.py')
appTest.run()


def test_init():
    assert not appTest.exception
    assert appTest.session_state['appData'] is None
    assert appTest.session_state['apiUrl'] == ''
    assert appTest.session_state['appId'] == ''
    assert appTest.session_state['curAppId'] is None
    assert appTest.session_state['score'] is None
    assert appTest.session_state['threshold'] is None
    assert appTest.session_state['explanationFig'] is None