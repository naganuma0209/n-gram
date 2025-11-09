# trigram.py
# Trigram モデルの最小実装

from collections import defaultdict


class TrigramModel:
    def __init__(self):
        # counts[(w1, w2)][w3] = 回数
        self.counts = defaultdict(lambda: defaultdict(int))

    def train(self, tokens):
        for w1, w2, w3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
            self.counts[(w1, w2)][w3] += 1

    def predict(self, w1, w2):
        """最も出現回数の多い w3 を返す"""
        next_words = self.counts[(w1, w2)]
        if not next_words:
            return None
        return max(next_words, key=next_words.get)

    def get_probability(self, w1, w2, w3):
        """Laplace smoothing なしの素朴な確率"""
        total = sum(self.counts[(w1, w2)].values())
        if total == 0:
            return 0.0
        return self.counts[(w1, w2)][w3] / total


if __name__ == "__main__":
    # テスト用
    with open("sample.txt", "r", encoding="utf-8") as f:
        tokens = f.read().split()

    model = TrigramModel()
    model.train(tokens)

    print("next word of ('I', 'love'):", model.predict("I", "love"))
