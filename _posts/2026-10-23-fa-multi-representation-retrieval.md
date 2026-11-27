---
layout: article
lang: fa
title: "چرا برای جست‌وجوی یک سند به چند embedding نیاز داریم؟"
description: "چند representation برای title، abstract و chunk را از multi-vectorهای ColBERT جدا می‌کنیم و trade-off حافظه، ranking و latency را توضیح می‌دهیم."
date: 2026-10-23 11:06:01 +0000
permalink: /fa/writing/multi-representation-retrieval/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "محمد شجاعی"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# چرا برای جست‌وجوی یک سند به چند embedding نیاز داریم؟

یک embedding برای کل سند ممکن است عنوان کوتاه یا نام خاصی را که فقط یک‌بار آمده زیر سایهٔ متن طولانی پنهان کند. گاهی بهتر است نمایه‌های مختلف را جدا نگه داریم: یکی برای عنوان، یکی برای چکیده و چند مورد برای بخش‌های متن. بعد می‌توانیم جست‌وجو را روی هرکدام انجام دهیم و نتیجه‌ها را دوباره ترکیب کنیم.

در Qdrant می‌توان برای یک سند چند بازنمایی متنی با سطح جزئیات متفاوت نگه داشت و هنگام retrieval آن‌ها را جداگانه جست‌وجو کرد. این طرح را باید از معنای دیگری از multi-vector هم جدا کرد.

## دو اصطلاح شبیه، دو طراحی متفاوت

**چند representation برای یک سند** یعنی برای بخش‌های از پیش تعریف‌شدهٔ سند چند embedding مستقل بسازیم: مثلاً title، abstract و body chunks. در زمان query، هر representation می‌تواند به نوع متفاوتی از سؤال جواب دهد.

روش **multi-vector با late interaction** یعنی مدل برای یک query یا passage چند بردار، مثلاً بردارهای token-level، تولید می‌کند و رتبه‌بندی روی تعامل چند بردار انجام می‌شود. ColBERT یک نمونهٔ شناخته‌شده از این رویکرد است. این طراحی محاسبات و index خاص خود را دارد و با ذخیرهٔ یک بردار برای عنوان و یکی برای متن یکی نیست.

در این طراحی، منظور representationهای جداگانه برای بخش‌های سند و ترکیب dense با BM25 است؛ این دو را نباید با همهٔ تعریف‌های multi-vector یکی گرفت.

## یک جست‌وجوی چندنمایشی

فرض کنید کاربر نام مقاله را دقیق می‌داند ولی واژه‌های پرسش با متن بدنه فرق دارند. جست‌وجوی lexical روی عنوان می‌تواند این نام خاص را پیدا کند؛ dense retrieval روی چکیده یا chunk ممکن است با یک توضیح معنایی به پاسخ برسد. سیستم هر retrieval leg را جدا اجرا می‌کند، سپس رتبه‌ها را با روشی مثل Reciprocal Rank Fusion ترکیب می‌کند و chunkهای برتر را به سند اصلی گروه می‌زند.

در tutorial مربوط به [Multi-Representation Search](https://qdrant.tech/documentation/tutorials-search-engineering/multi-representation-search/)، Qdrant نمونه‌ای از title، abstract و chunk را جدا نگه می‌دارد و result listها را fuse می‌کند. انتخاب RRF به این درد می‌خورد که scoreهای BM25 و embedding الزاماً در یک scale نیستند؛ RRF رتبه را ترکیب می‌کند، نه مقدار خام score را.

## هزینهٔ طراحی

هر representation جدید storage، indexing و گاهی محاسبهٔ بیشتر می‌خواهد. اگر عنوان و چکیده را برای هر chunk تکرار کنیم، سادگی یک collection یا query را با مصرف حافظه عوض کرده‌ایم. اگر جداگانه نگه داریم، باید رابطهٔ chunk، سند والد و نتایج را مدیریت کنیم. Fusion هم tuning می‌خواهد؛ اگر representation ضعیف باشد، اضافه‌کردنش می‌تواند ranking را خراب کند.

قبل از انتخاب schema، queryها را دسته‌بندی کنید: آیا کاربران نام و کد دقیق جست‌وجو می‌کنند؟ سؤال دربارهٔ کل سند می‌پرسند یا یک بخش خاص؟ بعد هر retrieval leg را جداگانه و همراه با fusion ارزیابی کنید. Log کنید هر leg چه چیزی را پیدا کرده؛ وگرنه معلوم نمی‌شود نتیجهٔ خوب از کدام representation آمده یا miss در کجا رخ داده است.

## چطور انتخاب کنیم؟

برای مجموعهٔ کوچک یا جست‌وجوی ساده، یک embedding برای chunkها ممکن است کافی باشد. اگر title یا abstract سیگنال مهمی است، آن‌ها را جدا امتحان کنید. اگر لازم است تطبیق ریزدانه میان query tokenها و passage tokenها انجام شود، late interaction مانند ColBERT را ارزیابی کنید؛ با این حال هزینهٔ حافظه و query-time را هم حساب کنید.

هیچ‌کدام را فقط به‌خاطر نام «multi-vector» انتخاب نکنید. معیار، بهترشدن queryهای هدف در برابر storage، latency و پیچیدگی نگه‌داری است.

## جمع‌بندی

چند representation به سیستم اجازه می‌دهد عنوان، چکیده و متن را با سیگنال‌های مناسب خودشان جست‌وجو کند. این قابلیت برای اسناد پژوهشی و corpusهایی با اسم خاص می‌تواند مفید باشد، اما یک نمایش اضافه رایگان نیست. retrieval legها را جدا اندازه بگیرید، نتایج را با روش روشن fuse کنید و بین جست‌وجوی چندنمایشی و late-interaction multi-vector تفاوت بگذارید.

## منابع
- **منبع:** [Multi-Representation Search، Qdrant](https://qdrant.tech/documentation/tutorials-search-engineering/multi-representation-search/)
- **منبع:** Khattab و Zaharia، [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832).
