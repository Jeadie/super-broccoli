from bottle import Bottle

app = Bottle()

@app.route('/')
def home():
    return 'You are a Driver!'


if __name__ == '__main__':
    app.run()
