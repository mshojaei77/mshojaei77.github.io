---
layout: article
lang: fa
title: "چطور Embedding Model مناسب برای فارسی انتخاب کنیم؟"
description: "FaMTEB و PTEB چه taskهایی را می‌سنجند و چطور با query set فارسی خودمان مدل را بر اساس retrieval، reranking، license و latency انتخاب کنیم؟"
date: 2025-07-15 16:38:08 +0000
permalink: /fa/writing/persian-embedding-models/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "محمد شجاعی"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
source_url: "https://t.me/LLMEngineers/15"
---

# چطور Embedding Model مناسب برای فارسی انتخاب کنیم؟

برای RAG فارسی دنبال یک «بهترین embedding model» همیشگی نگردید. مدلی که روی similarity خوب است ممکن است برای retrieval یا reranking انتخاب اول نباشد؛ مدل چندزبانه شاید بهتر از مدل فارسی‌محور به corpus خاص شما بخورد، یا برعکس. انتخاب را از task و query set شروع کنید.

در [پست کانال دربارهٔ PTEB](https://t.me/LLMEngineers/15) به leaderboard فارسی اشاره کردم. آن زمان PTEB راهی بود برای مقایسهٔ Text Embeddingها روی چند task؛ از آن زمان benchmarkهای فارسی دیگری هم اضافه شده‌اند.

## دو benchmark فارسی چه taskهایی دارند؟

در FaMTEB، مقالهٔ ۲۰۲۵ شصت‌وسه dataset در هفت task معرفی می‌کند: classification، clustering، pair classification، reranking، retrieval، summary retrieval و semantic textual similarity. مقاله ترکیبی از datasetهای موجود، ترجمه‌شده و تازه‌ساخته را توضیح می‌دهد. این گستردگی کمک می‌کند مدل را در چند نقش ببینیم؛ ولی score میانگین باز هم شاید ضعف یک task مهم برای محصول شما را پنهان کند.

در PTEB leaderboard نیز taskهایی مثل pair classification، retrieval، reranking، intent و sentiment کنار مشخصاتی مانند ابعاد embedding، context size و license آمده‌اند. اعداد leaderboard از revision و تنظیم همان ارزیابی می‌آیند؛ leaderboard زنده، snapshot ثابت مقاله نیست. برای تصمیم، revision، نحوهٔ encode کردن query/document و dataset را ثبت کنید.

## ابتدا مسئله را مشخص کنید

- **جست‌وجوی معنایی:** query کوتاه باید متن مرتبط را پیدا کند. Recall@k و nDCG را روی queryهای واقعی بسنجید.
- **پیداکردن عبارت دقیق:** نام، کد یا واژهٔ خاص مهم است؛ embedding را با BM25 یا hybrid retrieval مقایسه کنید.
- **رتبه‌بندی دوباره:** candidateها از قبل آمده‌اند و مدل باید بهترین‌ها را جلو بیاورد؛ reranking benchmark را جدا ببینید.
- **تشابه جمله‌ها یا دسته‌بندی:** pair classification و STS این کاربردها را بهتر نشان می‌دهند تا فقط retrieval.
- **پرسش فارسی، اسناد انگلیسی یا برعکس:** توانایی cross-lingual را با همان جهت query/document آزمایش کنید.

## یک راه انتخاب عملی

۱. ۱۰۰ تا ۳۰۰ query نماینده جمع کنید، روی سؤال‌های واقعی یا تست‌های بازبینی‌شده؛ این عدد نقطهٔ شروع داخلی است، نه حداقل علمی.
۲. برای هر query سند مرتبط و درجهٔ relevance را علامت بزنید و queryهای بدون جواب را هم نگه دارید.
۳. یک baseline واژگانی، یک multilingual embedding و یک فارسی‌محور را مقایسه کنید.
۴. روش ساخت بردار، pooling، normalization و similarity metric را مطابق model card اجرا کنید.
۵. برای هر query، Recall@k، nDCG، latency، حافظه و هزینه را ثبت کنید؛ بعد نمونه‌های miss را ببینید.
۶. اگر مدل تجاری است، retention و استفاده از داده را بررسی کنید؛ اگر self-host است، memory، سرعت و license مدل را هم بسنجید.

در فارسی، normalization را ثابت نگه دارید. شکل‌های ی و ک، نیم‌فاصله، رقم فارسی/لاتین و punctuation می‌توانند retrieval را عوض کنند. متن اصلی را دست‌نخورده ذخیره کنید و اگر normalization می‌آزمایید، نسخهٔ جدا و یکسان برای query و document بسازید.

## چرا benchmark عمومی به‌تنهایی کافی نیست؟

رقابت‌های فارسی کیفیت داده و taskهای متفاوتی دارند. ممکن است queryهای leaderboard شما را در بخش retrieval پوشش ندهند یا corpus حقوقی/فاکتور با متن‌های آموزشی آن فرق کند. همچنین translationese، queryهای مصنوعی و contamination می‌توانند score را بالا ببرند؛ dataset باید برای زبان و سناریوی واقعی بازبینی شود.

اگر دو مدل نزدیک‌اند، تنها به mean score نگاه نکنید. خطاهای retrieval و license می‌توانند انتخاب را عوض کنند. مدلی با context size بزرگ‌تر الزاماً ranking بهتری نمی‌دهد؛ مدل کوچک‌تر با recall مشابه ممکن است برای سرویس پرترافیک مناسب‌تر باشد.

## جمع‌بندی

برای انتخاب مدل، FaMTEB و PTEB فهرست کوتاهی از candidateها و taskهای مرتبط فراهم می‌کنند؛ آن‌ها answer نهایی محصول شما نیستند. مدلی را بردارید که روی query و سند فارسی خودتان، با setting درست، retrieval و پاسخ را بهتر می‌کند و از نظر هزینه، latency، license و privacy قابل‌استقرار است.

## منابع

- [پست کانال: PTEB Leaderboard](https://t.me/LLMEngineers/15)
- **منبع:** Zinvandi و همکاران، [FaMTEB: Massive Text Embedding Benchmark in Persian Language](https://arxiv.org/abs/2502.11571).
- **منبع:** [PTEB Leaderboard و داده‌های مدل](https://huggingface.co/spaces/opll-org/pteb-leaderboard)
- **منبع:** [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)
