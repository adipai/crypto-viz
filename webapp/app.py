"""
This script runs the app on the development server
"""
from os import environ,system
from cryptoviz import flaskapp
#from gevent.pywsgi import WSGIServer

if __name__ == "__main__":
    #http_server = WSGIServer(('0.0.0.0', 5000), flaskapp)
    #http_server.serve_forever()
    flaskapp.run(host='0.0.0.0')
