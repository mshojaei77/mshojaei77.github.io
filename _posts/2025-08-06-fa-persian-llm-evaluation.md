---
layout: article
lang: fa
title: "چطور توانایی واقعی مدل‌های زبانی را در فارسی بسنجیم؟"
description: "دانش، استدلال، multimodal، روانی کلام، فرهنگ و safety یک مهارت واحد نیستند؛ benchmarkهای فارسی را بر اساس سؤال واقعی انتخاب می‌کنیم."
date: 2025-08-06 15:47:14 +0000
permalink: /fa/writing/persian-llm-evaluation/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "محمد شجاعی"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
source_url: "https://t.me/LLMEngineers/134"
---

# چطور توانایی واقعی مدل‌های زبانی را در فارسی بسنجیم؟

یک مدل ممکن است سؤال دانشی فارسی را درست جواب دهد، اما در مکالمه مصنوعی حرف بزند، تصویر را غلط بخواند یا درخواست حساس را ناامن پاسخ دهد. برای همین یک leaderboard یا benchmark به‌تنهایی جواب «کدام مدل فارسی را بلد است؟» را نمی‌دهد؛ باید مشخص کنیم چه توانایی‌ای را برای کدام محصول می‌سنجیم.

در [پست معرفی ParsiEval](https://t.me/LLMEngineers/134)، [آزمایش روی Konkur 1404](https://t.me/LLMEngineers/310) و [ارزیابی روانی کلام](https://t.me/LLMEngineers/313) من چند لایهٔ متفاوت را بررسی کردم.

## دانش و فهم زبان

در ParsiEval، من ۳۶۴ سؤال چندگزینه‌ای دربارهٔ تاریخ، ادبیات، دانش عمومی و علم ساختم تا baselineای برای دانش و استدلال فارسی داشته باشم. نتیجهٔ پست اولیه در آگوست ۲۰۲۵، GPT-4o را حدود ۷۹٪ و Kimi K2 را حدود ۷۷٪ گزارش می‌کرد. این score یک snapshot تاریخی از آن dataset و run است، نه رتبه‌بندی امروز.

در [repository فعلی ParsiEval](https://github.com/mshojaei77/ParsiEval)، benchmarkها در یک suite جمع شده‌اند. Konkur 1404، امتحان ورودی دانشگاه سال ۱۴۰۴ را می‌سنجد: ۲٬۱۳۷ سؤال فارسی و انگلیسی در ده session، با پشتیبانی از شکل و تصویر. این مجموعه علاوه بر دانش، محاسبه و استدلال را آزمایش می‌کند؛ اما نمرهٔ امتحان، معادل کیفیت chatbot در کار روزانه نیست.

## زبان روان و register

مدل می‌تواند پاسخ درست بدهد ولی ترجمه‌وار، بیش‌ازحد رسمی یا ناآشنا برای مکالمهٔ ایرانی بنویسد. در پست اولیه، Persian Linguistic Fluency را با معیارهایی مثل دستور، طبیعی‌بودن و idiomها توضیح دادم و از Gemini به‌عنوان judge استفاده کرده بودم. ارزیابی judge خودکار را باید همراه prompt، مدل judge و روش کالیبراسیون گزارش کرد؛ یک مدل دیگر هم ممکن است slang یا register را متفاوت بفهمد.

در نسخهٔ فعلی repository، Persian Fluency Bench شامل ۳۰ case داوری‌شده در ۱۰ domain روزمره و دو track رسمی و محاورهٔ تهرانی است. این bench یک بخش خاص را می‌سنجد؛ آن را با دانش، reasoning یا safety در یک score جمع نکنید مگر ترکیب و وزن‌دهی را دقیقاً توضیح دهید.

## ایمنی، انصاف و هنجارهای فرهنگی

روانی فارسی به‌تنهایی رفتار ایمن یا منصفانه را تضمین نمی‌کند. [ELAB](https://arxiv.org/abs/2504.12553) ارزیابی فارسی را در سه بعد safety، fairness و social norms بررسی می‌کند و benchmarkهای ویژهٔ این ابعاد را معرفی می‌کند. [PERCUL](https://aclanthology.org/2025.naacl-long.631/) روی سناریوهای فرهنگی و هنجارهای اجتماعی ایران متمرکز است. این‌ها سؤال متفاوتی از ParsiEval یا Konkur می‌پرسند و نباید به یک leaderboard عمومی تقلیل داده شوند.

برای هر حوزه باید معلوم باشد پاسخ مطلوب را چه کسی تعریف کرده، چطور اختلاف ارزیاب حل شده و چه نوع failure سنجیده می‌شود. مثلاً refusal در یک آزمون safety با روانی گفتار یا احترام به taarof یک معیار واحد نیست.

## توانایی‌های multimodal و دنیای واقعی

در Konkur 1404 امکان ارزیابی سؤال‌های تصویری و هندسی هم هست. در پست فنی دربارهٔ همین leaderboard نوشتم که text-only score با standard score می‌تواند فرق کند؛ این یادآوری می‌کند که وقتی محصول PDF، نمودار یا تصویر دریافت می‌کند، benchmark متنی کافی نیست.

اگر سیستم قرار است سند بخواند، OCR و parsing را جداگانه بسنجید؛ اگر intent را route می‌کند، confusion matrix هر دسته را ببینید؛ و اگر پاسخ حساس می‌دهد، abstention و ارجاع به انسان را هم بسنجید. failure در محصول ممکن است از preprocessing باشد، نه خود مدل زبانی.

## یک پروتکل ارزیابی قابل‌دفاع

۱. **تصمیم محصول را مشخص کنید.** مدل باید چه کاری برای کاربر انجام دهد؟
۲. **مهارت‌ها را جدا کنید.** دانش، استدلال، زبان طبیعی، دیدن تصویر، safety و tool use آزمون‌های جدا می‌خواهند.
۳. **مجموعه‌داده و معیار را مستند کنید.** نسخه، منبع نمونه‌ها، روش ترجمه، split و معیار اصلی را بنویسید.
۴. **خط پایه و uncertainty را گزارش کنید.** مدل ساده، اجرای تکراری، خطای هر دسته و اندازهٔ نمونه را کنار score بیاورید.
۵. **نمونه‌های خطا را بازبینی کنید.** پاسخ غلط چه الگویی دارد و چه سطحی از ریسک ایجاد می‌کند؟
۶. **نسخه‌ها را نگه دارید.** model ID، prompt، decoding، tool و تاریخ اجرا بخشی از نتیجه‌اند.

اگر dataset عمومی است، احتمال contamination را هم در نظر بگیرید. اگر judge مدل زبانی است، agreement آن با native-speaker annotators و حساسیت به prompt را بسنجید. اگر benchmark تازه ساخته‌اید، ابتدا آن را pilot بنامید تا زمانی که روش و labels ارزیابی مستقل داشته باشند.

## جمع‌بندی

ارزیابی مدل فارسی یک عدد نیست. برای انتخاب model، benchmark را با کار واقعی هم‌راستا کنید: intent، knowledge، fluency، multimodal، safety و هنجارهای فرهنگی را جدا بسنجید و بعد روی دادهٔ محصول خودتان retest کنید. benchmark عمومی مسیر انتخاب را کوتاه می‌کند؛ تصمیم نهایی باید بر اساس failureهایی باشد که برای کاربر شما مهم‌اند.

## منابع

- [پست کانال: ParsiEval](https://t.me/LLMEngineers/134)
- [پست کانال: Konkur 1404](https://t.me/LLMEngineers/310)
- [پست کانال: Persian Linguistic Fluency](https://t.me/LLMEngineers/313)
- **منبع:** [ParsiEval repository](https://github.com/mshojaei77/ParsiEval)
- **منبع:** [ELAB: Extensive LLM Alignment Benchmark in Persian](https://arxiv.org/abs/2504.12553)
- **منبع:** [PERCUL: A Story-Driven Cultural Evaluation of LLMs in Persian](https://aclanthology.org/2025.naacl-long.631/)
