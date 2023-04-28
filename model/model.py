import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TextVectorization

from model.positional_embedding import PositionalEmbedding
from model.transformer_decoder import TransformerDecoder


class LangModel:
    VOCAB_SIZE = 15000
    SEQUENCE_LENGTH = 100

    TEXT_VECTORIZATION = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )

    TOKENS_INDEX = {}

    def generate(self, sentence: str, length: int):
        for i in range(length):
            tokenized_sentence = LangModel.TEXT_VECTORIZATION([sentence])
            predictions = self.model(tokenized_sentence)
            next_token = LangModel.sample_next(predictions[0, i, :])
            sampled_token = LangModel.TOKENS_INDEX[next_token]
            sentence += " " + sampled_token
        return sentence

    @staticmethod
    def __prepare_lm_dataset(text_batch):
        vectorized_sequences = LangModel.TEXT_VECTORIZATION(text_batch)
        x = vectorized_sequences[:, :-1]
        y = vectorized_sequences[:, 1:]
        return x, y

    @staticmethod
    def sample_next(predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype("float64")
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)

    def __init__(self, embed_dim, latent_dim, num_heads, vocab_size, sequence_length):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dataset = tf.keras.utils.text_dataset_from_directory(
            directory="dataset/aclImdb", label_mode=None, batch_size=16)
        self.dataset = self.dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        LangModel.TEXT_VECTORIZATION.adapt(self.dataset)
        LangModel.TOKENS_INDEX = dict(enumerate(LangModel.TEXT_VECTORIZATION.get_vocabulary()))
        inputs = Input(shape=(None,), dtype="int64")
        x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
        x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
        outputs = Dense(vocab_size, activation="softmax")(x)
        self.model = Model(inputs, outputs)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")

    def fit(self):
        prompt = "This movie"
        text_gen_callback = TextGenerator(
            prompt,
            generate_length=50,
            model_input_length=self.sequence_length,
            temperatures=(0.2, 0.5, 0.7, 1., 1.5))
        lm_dataset = self.dataset.map(Model.__prepare_lm_dataset, num_parallel_calls=4)
        self.model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])

    def load_weights(self, path: str):
        self.model.load_weights(path)


class TextGenerator(Callback):
    def __init__(self,
                 prompt,
                 generate_length,
                 model_input_length,
                 temperatures=(1.,),
                 print_freq=1):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        for temperature in self.temperatures:
            print("== Generating with temperature", temperature)
            sentence = self.prompt
            for i in range(self.generate_length):
                tokenized_sentence = LangModel.TEXT_VECTORIZATION([sentence])
                predictions = self.model(tokenized_sentence)
                next_token = LangModel.sample_next(predictions[0, i, :])
                sampled_token = LangModel.TOKENS_INDEX[next_token]
                sentence += " " + sampled_token
            print(sentence)
