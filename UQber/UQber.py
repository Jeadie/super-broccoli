from bottle import Bottle, get, post, request
from Passenger import Passenger
from Driver import Driver
app = Bottle(__name__)
app.mount("/Passenger",Passenger.app )
app.mount("/Driver", Driver.app)


@app.post('/')
def user_redirect():
    if request.forms.get("User") == ""

@app.get('/')
def home():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
