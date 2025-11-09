# smoothing.py
# Laplace smoothing + Bigram の実装例

from collections import defaultdict


class BigramWithSmoothing:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, tokens):
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            self.counts[w1][w2] += 1
            self.vocab.add(w1)
            self.vocab.add(w2)

    def get_probability(self, w1, w2):
        """Laplace smoothing の確率を返す"""
        vocab_size = len(self.vocab)
        count_w1_w2 = self.counts[w1][w2]
        total_w1 = sum(self.counts[w1].values())

        # Laplace smoothing
        return (count_w1_w2 + 1) / (total_w1 + vocab_size)

    def predict(self, w1):
        """スムージングを使った予測"""
        vocab = list(self.vocab)
        probabilities = {w2: self.get_probability(w1, w2) for w2 in vocab}
        return max(probabilities, key=probabilities.get)


if __name__ == "__main__":
    # テスト
    with open("sample.txt", "r", encoding="utf-8") as f:
        tokens = f.read().split()

    model = BigramWithSmoothing()
    model.train(tokens)

    print("next word of 'I' (with smoothing):", model.predict("I"))
