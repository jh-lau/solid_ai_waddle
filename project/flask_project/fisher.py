"""
  @Author       : liujianhan
  @Date         : 21/3/18 23:17
  @Project      : solid_ai_waddle
  @FileName     : fisher.py
  @Description  : Placeholder
"""
from flask import Flask

app = Flask(__name__)


@app.route('/hello')
def hello():
    return "hello fisher"


if __name__ == '__main__':
    app.run(debug=True)
