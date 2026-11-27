---
layout: article
lang: fa
title: "از AdamW تا Muon؛ مقایسهٔ optimizerها در آموزش LLM"
description: "تفاوت AdamW و Muon را از روی نوع update توضیح می‌دهیم و ادعاهای سرعت را با هزینهٔ واقعی، تنظیم hyperparameter و طراحی آزمایش می‌سنجیم."
date: 2025-07-14 16:15:33 +0000
permalink: /fa/writing/muon-vs-adamw/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "محمد شجاعی"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
source_url: "https://t.me/LLMEngineers/5"
---

# از AdamW تا Muon؛ مقایسهٔ optimizerها در آموزش LLM

بهینه‌ساز (optimizer) با gradientها وزن‌ها را به‌روزرسانی می‌کند تا loss پایین بیاید. در آموزش زبان، همین انتخاب می‌تواند روی تعداد گام‌ها، حافظهٔ optimizer، پایداری و زمان رسیدن به کیفیت هدف اثر بگذارد. پست [کانال دربارهٔ Muon](https://t.me/LLMEngineers/5) نتیجهٔ یک paper را معرفی می‌کرد که نشان داده Muon در آزمایش‌های خاص می‌تواند با FLOPs کمتر به loss مشابه برسد. برای تفسیر آن نتیجه باید بدانیم Muon چه تغییری می‌دهد و مقایسه چطور انجام شده است.

## بهینه‌ساز AdamW چه می‌کند؟

در **AdamW** برای هر پارامتر میانگین متحرک gradient و میانگین متحرک مربع gradient نگه داشته می‌شود. این دو مقدار update را با توجه به اندازهٔ gradientهای اخیر مقیاس می‌کنند. AdamW همچنین weight decay را از update تطبیقی جدا می‌کند؛ این تفاوت در [مقالهٔ Decoupled Weight Decay](https://arxiv.org/abs/1711.05101) صورت‌بندی شده است.

مزیت عملی AdamW فقط معادله نیست. پیاده‌سازی‌ها، recipeها و تجربهٔ تیمی زیادی برای آن وجود دارد. از طرف دیگر، نگهداری state برای هر پارامتر حافظه می‌خواهد و رفتار الگوریتم به learning rate، weight decay، schedule و batch حساس است.

## ایدهٔ Muon

در روش **Muon**، gradient یا momentum ماتریس‌های وزن با روش‌هایی از خانوادهٔ matrix orthogonalization تبدیل می‌شود تا جهت update ساختار دیگری پیدا کند. در پیاده‌سازی‌های رایج، این تبدیل با تقریب‌هایی مانند Newton–Schulz انجام می‌شود. هدف، بهره‌بردن از اطلاعات ساختاری ماتریس‌هاست؛ نه این‌که هر پارامتر با یک فرمول یکسان و بدون تنظیم بهینه شود.

مقالهٔ [Muon Is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) برای مقیاس‌دادن Muon راهکارهایی مثل weight decay و تنظیم مقیاس update را بررسی می‌کند و در scaling law آزمایش‌شده، حدود دو برابر کارایی محاسباتی نسبت به baseline گزارش می‌دهد. این عدد نتیجهٔ setup همان مقاله است؛ به معنی نصف‌شدن زمان یا هزینهٔ هر training run در هر سخت‌افزار و مدل نیست.

در بسیاری از training recipeها، Muon فقط برای بعضی وزن‌های دوبعدی به‌کار می‌رود و پارامترهایی مثل embedding، bias یا normalization ممکن است optimizer دیگری داشته باشند. جزئیات پیاده‌سازی را بخوانید؛ واژهٔ «Muon training» به‌تنهایی نمی‌گوید کدام پارامترها با چه الگوریتمی به‌روزرسانی شده‌اند.

## چرا یک optimizer لزوماً برندهٔ عمومی نیست؟

سرعت optimizer را نمی‌شود فقط با FLOPs محاسبه کرد. اگر یک روش هر گام سریع‌تر باشد ولی به گام‌های بیشتری برسد، نتیجهٔ wall-clock ممکن است متفاوت شود. ارتباط بین communication، kernel، batch، اندازهٔ مدل، شبکهٔ GPU و memory bandwidth نیز روی زمان اثر می‌گذارد. optimizer می‌تواند state حافظه را کمتر کند، اما اگر implementation توزیع‌شده کند باشد، نتیجهٔ end-to-end بهتر نخواهد شد.

پژوهش‌های بعدی بر اهمیت مقایسهٔ منصفانه تأکید دارند. [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) نشان می‌دهد تنظیم نابرابر hyperparameterها و معیارهای محدود می‌تواند ادعای speedup را مبهم کند. مطالعهٔ [Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222) و بررسی چند میلیارد پارامتری [SOAP, Muon, and Beyond](https://arxiv.org/abs/2607.20548) نیز یافته‌ها را در setupهای مشخص خود گزارش می‌کنند. این مطالعات دلیل خوبی برای آزمایش کنترل‌شده‌اند، نه برای جایگزینی یک پیش‌فرض قطعی با پیش‌فرضی دیگر.

## مقایسهٔ کاربردی چطور طراحی شود؟

اگر در حال آموزش مدلی هستید، از مدل کوچک‌تر یا آزمایش ابلیشن شروع کنید. برای هر optimizer، بودجهٔ تنظیم تقریباً برابر، دادهٔ ثابت و معیار نهایی یکسان داشته باشید. دست‌کم این‌ها را ثبت کنید:

- **کیفیت:** validation loss و معیار task در checkpointهای هم‌بودجه.
- **هزینهٔ محاسباتی:** FLOPs یا tokenها تا رسیدن به کیفیت هدف.
- **زمان واقعی:** wall-clock همراه با نوع سخت‌افزار و پیکربندی parallelism.
- **حافظه:** model state، optimizer state، activation و peak memory.
- **پایداری:** loss spike، NaN، حساسیت به seed و نیاز به تنظیم مجدد.

برای مقایسهٔ هزینه، فقط loss نهایی یک checkpoint را با هم نسنجید. اگر یکی از روش‌ها بودجهٔ sweep بیشتری گرفته یا learning rateهای متفاوت داشته، برتری ممکن است از کیفیت tuning بیاید نه ذات optimizer.

## چه زمانی Muon را امتحان کنیم؟

روش Muon ارزش آزمایش دارد وقتی آموزش یا پژوهش شما اجازهٔ کنترل recipe و ثبت runها را می‌دهد، implementation مورد استفاده برای معماری‌تان پشتیبانی دارد و اندازه‌گیری می‌تواند هم کیفیت و هم زمان را پوشش دهد. اگر یک مدل آماده را فقط fine-tune می‌کنید یا از سرویس تجاری استفاده می‌کنید، optimizer ممکن است دسترس‌پذیر نباشد؛ در این حالت LoRA، QLoRA، داده و تنظیم training شاید تصمیم‌های اثرگذارتری باشند.

## جمع‌بندی

بهینه‌ساز AdamW هنوز baseline عملی مهمی است؛ Muon یک روش ماتریسی است که در برخی آزمایش‌های LLM به trade-off امیدوارکننده‌ای رسیده است. نتیجهٔ پژوهش را باید همراه با model size، data, compute, tuning و implementation بخوانیم. پرسش مفید این نیست که «کدام optimizer همیشه بهتر است؟» بلکه این است: «برای بودجه و کیفیت هدف من، کدام روش سریع‌تر و پایدارتر است؟»

## منابع

- [پست کانال: معرفی Muon](https://t.me/LLMEngineers/5)
- **منبع:** [Muon Is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- **منبع:** [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- **منبع:** [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046)
- **منبع:** [Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222)
- **منبع:** [SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales](https://arxiv.org/abs/2607.20548)
