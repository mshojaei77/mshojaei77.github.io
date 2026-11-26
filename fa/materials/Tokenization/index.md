---
layout: article
lang: fa
title: "آموزش Tokenization؛ تبدیل متن به توکن"
permalink: /fa/materials/Tokenization/
nav_exclude: true
---

# آموزش Tokenization؛ تبدیل متن به توکن

Tokenization یکی از مراحل مهم پردازش زبان است؛ متن را به توکن‌های قابل پردازش تبدیل می‌کند.

## چرا Tokenization مهم است؟
- نحوهٔ شکستن عددها به توکن ممکن است در مسائل حسابی اثر داشته باشد.
- عملیات روی رشته‌ها نیز به نحوهٔ توکن‌سازی وابسته است.

## Pipeline توکن‌سازی
1. Normalize: پاک‌سازی و استانداردسازی متن.
2. Pre-tokenize: شکستن متن به واژه یا زیرواژه.
3. Model: استفاده از BPE یا روشی مشابه برای تولید شناسه‌ها.
4. Post-process: افزودن توکن‌های ویژه مانند `[BOS]` و `[EOS]`.

## روش‌های توکن‌سازی
- سطح نویسه: عمومی، اما ممکن است دنبالهٔ طولانی ایجاد کند.
- سطح واژه: معنادار، اما با واژه‌های خارج از واژگان مشکل دارد.
- زیرواژه: BPE تعادلی برای کارایی ایجاد می‌کند.
- سطح بایت: استفاده از UTF-8 خام برای پشتیبانی چندزبانه.

## مراحل BPE
- آغاز با نویسه‌ها.
- شمارش فراوانی جفت‌ها.
- ادغام پرتکرارترین جفت‌ها.
- تکرار تا رسیدن به اندازهٔ واژگان موردنظر.

## یادداشت‌های من دربارهٔ Tokenization

- [Medium: Token Theory](https://medium.com/@mshojaei77/introduction-to-tokenization-a-theoretical-perspective-b1cc22fe98c5)
- [Colab: Tokenization Techniques](https://colab.research.google.com/drive/1RwrtINbHTPBSRIoW8Zn9BRabxXguRRf0?usp=sharing)

## منابع پیشنهادی Tokenization

- [Mistral Tokenization Guide](https://docs.mistral.ai/guides/tokenization/)
- [Hugging Face Pipeline](https://huggingface.co/docs/tokenizers/en/pipeline)
- [Hugging Face Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
- [Airbyte LLM Tokenization Guide](https://airbyte.com/data-engineering-resources/llm-tokenization)
- [Tiktokenizer](https://tiktokenizer.vercel.app/)
- [MIT Paper on Tokenization](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00557/128327)
- [Medium: BPE Tutorial](https://medium.com/@mshojaei77/understanding-bpe-tokenization-a-hands-on-tutorial-80570314b12f)
- [BPE Original Paper](https://arxiv.org/abs/1508.07909)
- [Karpathy Video on Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)
- [MinBPE GitHub Repo](https://github.com/karpathy/minbpe)

شما برای مدیریت واژه‌های نادر در توکن‌سازی چه روشی دارید؟

کلیدواژه‌ها: Tokenization، BPE، توکن‌سازی زیرواژه، تبدیل متن به توکن
