from flask import Flask, render_template, request

from model.model import LangModel

app = Flask(__name__)

model = LangModel(embed_dim=256, latent_dim=2048, num_heads=2, vocab_size=15000, sequence_length=100)

model.load_weights("my_model.h5")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form.get('text_input')
        length = request.form.get('integer_input')
        output = model.generate(sentence, length)
        return render_template('response.html', input=sentence, answer=output)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
