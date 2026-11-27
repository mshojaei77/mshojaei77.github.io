---
layout: article
lang: fa
title: "بازیابی Contextual؛ چطور context گم‌شدهٔ chunk را برگردانیم؟"
description: "روش contextual retrieval را با ایجاد context کوتاه در زمان ingestion توضیح می‌دهیم و نتایج Anthropic را از ادعای عمومی جدا می‌کنیم."
date: 2026-10-24 08:09:23 +0000
permalink: /fa/writing/contextual-retrieval/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "محمد شجاعی"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
source_url: "https://t.me/LLMEngineers/539"
---

# بازیابی Contextual؛ چطور context گم‌شدهٔ chunk را برگردانیم؟

ممکن است chunk دقیقاً جواب سؤال را داشته باشد، اما معلوم نباشد «این بخش» دربارهٔ کدام مشتری، قرارداد یا بازهٔ زمانی است. وقتی chunk جداگانه جست‌وجو می‌شود، عنوان و زمینهٔ سند اصلی از دید retriever پنهان می‌ماند. Contextual Retrieval می‌کوشد همین زمینهٔ گم‌شده را پیش از ساخت index به chunk اضافه کند.

این همان مشکلی است که در [پست کانال دربارهٔ Contextual Retrieval](https://t.me/LLMEngineers/539) توضیح دادم.

## روش چطور کار می‌کند؟

در زمان ingestion، مدل سند کامل و یک chunk را با هم می‌بیند و توضیح کوتاهی می‌سازد که آن بخش را در زمینه قرار می‌دهد. مثلاً می‌گوید این پاراگراف دربارهٔ بند فسخ قرارداد شمارهٔ فلان و نسخهٔ تاریخ مشخص است. سپس این context همراه خود chunk وارد dense embedding و جست‌وجوی واژگانی می‌شود.

وقتی کاربر query می‌فرستد، سیستم همچنان retrieval معمول را انجام می‌دهد؛ فقط متن‌های indexشده سرنخ بیشتری دارند. این روش با chunking تفاوت دارد: مرز chunk را عوض نمی‌کند، بلکه نشانه‌ای از سند اصلی به آن می‌چسباند. همچنین memory یا knowledge تازه برای مدل نمی‌سازد؛ فقط نمایش موردجست‌وجو را غنی می‌کند.

## نتایج گزارش‌شدهٔ Anthropic چیست؟

در مقالهٔ [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)، Anthropic ترکیب embedding، BM25، contextualization و reranking را روی مجموعه‌داده‌های خودش مقایسه کرد. روی معیار failure-to-retrieve@20، یعنی سهم سؤال‌هایی که پاسخ در ۲۰ نتیجهٔ اول پیدا نشد، مقاله کاهش از ۵٫۷٪ به ۲٫۹٪ را برای contextualization و به ۱٫۹٪ را برای ترکیب contextualization با reranking گزارش می‌کند.

این نتایج از آزمایش همان مجموعه و تنظیمات می‌آیند؛ تضمین نمی‌کنند پروژهٔ شما همین مقدار بهبود بگیرد. اگر chunkها از قبل عنوان و metadata کافی دارند، یا سؤال‌ها با عبارت‌های دقیق در متن پیدا می‌شوند، contextualization شاید هزینهٔ اضافی و فایدهٔ کمی داشته باشد.

## هزینه و failure modeها

برای هر chunk باید context بسازید؛ پس ingest طولانی‌تر و پرهزینه‌تر می‌شود و تغییر سند احتمالاً به بازسازی بخش‌های وابسته نیاز دارد. context تولیدشده هم ممکن است یک نام یا رابطه را اشتباه نسبت بدهد. اگر متن تولیدشده بی‌بررسی index شود، عملاً یک خطای تازه وارد retrieval کرده‌ایم.

نسخه و سند اصلی را حفظ کنید، context تولیدشده را به‌عنوان metadata مشتق‌شده نگه دارید و روی نمونه‌ای از chunkها آن را بررسی کنید. در evaluation، baseline بدون contextualization را کنار نسخهٔ جدید بگذارید. علاوه‌بر Recall@k، latency، حجم index و کیفیت پاسخ نهایی را مقایسه کنید.

رتبه‌بندی دوباره را جداگانه بسنجید. Contextualization در زمان ingestion هزینه دارد؛ reranker هزینهٔ query-time و latency اضافه می‌کند. اگر هر دو را هم‌زمان اضافه کنید و نتیجه بهتر شود، نمی‌دانید کدام مرحله مؤثر بوده است.

## چه زمانی آزمایشش ارزش دارد؟

اگر retrieval زیاد chunk درست را از دست می‌دهد چون بخش‌های متن بدون نام سند، عنوان، موضوع یا زمان قابل‌فهم نیستند، contextual retrieval گزینهٔ خوبی برای آزمایش است. اگر مشکل از parser، ACL، index کهنه یا query rewrite باشد، افزودن context مصنوعی ممکن است فقط failure اصلی را بپوشاند.

## جمع‌بندی

این روش زمینهٔ لازم را پیش از index به chunk اضافه می‌کند تا جست‌وجوی واژگانی و معنایی بتواند بهتر آن را پیدا کند. ایده به‌خصوص برای chunkهایی که بیرون از سند مبهم‌اند جذاب است، اما باید روی queryهای خودتان ارزیابی شود. اول علت miss را پیدا کنید؛ بعد این مرحله را با هزینه و metricهای روشن وارد pipeline کنید.

## منابع

- [پست کانال: Contextual Retrieval](https://t.me/LLMEngineers/539)
- **منبع:** [Contextual retrieval in AI systems، Anthropic](https://www.anthropic.com/research/contextual-retrieval)
