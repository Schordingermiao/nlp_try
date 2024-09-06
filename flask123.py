#!/usr/bin/env python3


from main4 import find_similar_data

from flask import Flask, redirect, url_for, abort, render_template,request
from flask import jsonify
import json
app = Flask(__name__)

app.json.ensure_ascii = False

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/',methods=['POST'])
def getAbstractProp():
    data = request.get_data()
    tem_data=data.decode('utf-8')
    tem_data=tem_data[9:len(tem_data)-1]
    re=find_similar_data(tem_data)["review"]

    jdata = re.to_json(orient='records')
    return jsonify(json.loads(jdata))




    return re

@app.route('/user/<username>')
def user_profile(username):
    return f'Hello, {username}!'


if __name__ == '__main__':
    with app.test_request_context():
        # 生成指向index视图函数的URL
        index_url = url_for('index')
        print(f'URL for index: {index_url}')	# URL for index: /

        # 生成指向user_profile视图函数的URL，传递参数username='admin'
        user_profile_url = url_for('user_profile', username='admin')
        print(f'URL for user_profile: {user_profile_url}')	# URL for user_profile: /user/admin

    app.run('0.0.0.0', 5000)

