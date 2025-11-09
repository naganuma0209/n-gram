# bigram.py
# Bigram モデルの最小実装

from collections import defaultdict


class BigramModel:
    def __init__(self):
        # counts[w1][w2] = 回数
        self.counts = defaultdict(lambda: defaultdict(int))

    def train(self, tokens):
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            self.counts[w1][w2] += 1

    def predict(self, w1):
        """最も出現回数の多い w2 を返す"""
        next_words = self.counts[w1]
        if not next_words:
            return None
        return max(next_words, key=next_words.get)

    def get_probability(self, w1, w2):
        """Laplace smoothing なしの素朴な確率"""
        total = sum(self.counts[w1].values())
        if total == 0:
            return 0.0
        return self.counts[w1][w2] / total


if __name__ == "__main__":
    # テスト用
    with open("sample.txt", "r", encoding="utf-8") as f:
        tokens = f.read().split()

    model = BigramModel()
    model.train(tokens)

    print("next word of 'I':", model.predict("I"))
