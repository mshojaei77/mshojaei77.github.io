---
layout: article
lang: fa
title: "مدل‌های زبانی سنتی؛ پیش از Transformer"
permalink: /fa/materials/Traditional_LMs/
nav_exclude: true
---

# مدل‌های زبانی سنتی؛ پیش از Transformer

مدل‌های زبانی از n-gram تا RNN، پایه‌ای برای فهم مدل‌های امروزی‌اند.

## مدل n-gram
- شمارش واژه‌های قبلی برای پیش‌بینی واژهٔ بعدی.
- فرض مارکوف: محدودکردن تاریخچه برای کارایی.
- هموارسازی مانند Laplace برای مدیریت شمارش صفر.

## مدل زبانی عصبی Feedforward
- شبکهٔ عصبی با پنجرهٔ ثابت.
- Embedding و MLP برای پیش‌بینی احتمال.

## RNN در مدل‌سازی زبان
- LSTM و GRU برای مدیریت وابستگی‌های طولانی.
- مبانی Seq2seq برای وظایف پیشرفته.

## چرا مدل‌های سنتی را مطالعه کنیم؟
برای فهم مسیر تحول تا Transformer مفیدند. نظر شما دربارهٔ تفاوت RNN و Transformer چیست؟

## یادداشت‌های من دربارهٔ مدل‌های سنتی

- [Medium: N-Gram Basics](https://medium.com/@mshojaei77/understanding-n-gram-language-models-a-building-block-of-modern-nlp-3e4f3f0e1d3d)
- [Colab: N-Gram Playground](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)

## منابع پیشنهادی مدل‌های سنتی

- [Jurafsky SLP Book](https://web.stanford.edu/~jurafsky/slp3/)
- [Karpathy on RNN Effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [LSTM Original Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU Paper](https://arxiv.org/abs/1412.3555)
- [Colah's LSTM Post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Yoav Goldberg Primer](https://arxiv.org/abs/1807.10854)

کلیدواژه‌ها: مدل زبانی سنتی، n-gram، RNN، پیش‌بینی واژه
