from flask import Flask, request, render_template
from ai_model import ai


app = Flask(__name__)



@app.route('/movies', methods=['GET', 'POST'])
def main():
    # print(request.args.get('title'))
    # res = ai.results(request.args.get('title'))
    #res = ai.results(name)

    if request.method == 'GET':
        return render_template('index.htm')

    if request.method == 'POST':
        movie_name = request.form['movie_name']
        movie = str(movie_name)
        res = ai.results(movie)
        if type(res) != str:
            return render_template('recommendations.htm', movies=res)
        else: 
            return render_template('404.htm')
            
            

   

if __name__ == "__main__":
    app.run(port = 5000, debug=True)