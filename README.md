# 📘 N-gram Language Model Studies

このリポジトリは、自然言語処理の基礎的なモデルである **n-gram 言語モデル** を学習・実験した内容をまとめたものです。  
n-gram の数学的背景、計算方法、Pythonによる実装例、スムージングの扱いなどを整理しながら学習したプロセスを公開しています。

---

## 🎯 目的

- n-gram モデルの基礎を体系的に理解する
- Python で最小構成の n-gram モデルを実装する
- 言語モデルの歴史的背景（NNLM → word2vec → Transformer）へ続く理解の土台を作る

---

## 🧠 N-gram モデルとは？

n-gram モデルは、「直前の n−1 個の単語から次の単語の確率を推定する」シンプルな確率モデルです。

例：  
- **unigram（1-gram）**: 単語単独の出現確率  
- **bigram（2-gram）**: 直前の1単語から次の単語を予測  
- **trigram（3-gram）**: 直前の2単語から次の単語を予測  

形式的には：

P(w_n | w_{n-1}, ..., w_{n-(n-1)})

yaml
コードをコピーする

---

## ✏️ 学習メモ（要点）

- n-gram は「統計的言語モデル」の典型例
- 単純だが大型データでは驚くほど強い
- Pythonで簡単に実装できる
- ただし**sparseness（疎なデータ問題）**が深刻
- Laplace smoothing / Kneser-Ney smoothingなどの改善が必要
- ディープラーニング以前の言語モデルの基盤となっていた

---

## 🧪 Python 実装（最小サンプル）

このリポジトリには以下のファイルが含まれています：

ngram/
├── bigram.py # Bigram モデルの実装
├── trigram.py # Trigram モデルの実装
├── smoothing.py # ラプラススムージングの実装
└── sample.txt # 学習に用いたサンプル文章

python
コードをコピーする

最小構成の bigram のコード例：

```python
from collections import defaultdict

class BigramModel:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def train(self, tokens):
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            self.counts[w1][w2] += 1

    def predict(self, w1):
        next_words = self.counts[w1]
        if not next_words:
            return None
        return max(next_words, key=next_words.get)
🚀 実行方法
nginx
コードをコピーする
python bigram.py
または

nginx
コードをコピーする
python trigram.py
📄 参考文献
Jurafsky & Martin, “Speech and Language Processing”

Shannon, C. (1951). Prediction and Entropy of Printed English.

Bengio et al., “A Neural Probabilistic Language Model” (2003)

📝 今後の拡張予定
Kneser-Ney smoothing の実装

n を可変にした汎用 n-gram モデル

文生成サンプル（テキスト生成）

n-gram と word2vec の比較

✨ 作者メモ
このリポジトリは、言語モデルの基礎理解のための学習ログです。
まずは統計的な n-gram から始まり、後に word2vec / GloVe / FastText / Transformer などに続く予定です。