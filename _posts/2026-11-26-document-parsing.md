---
layout: article
lang: fa
article_kind: essay
show_article_header: false
title: "مهندسی Document Parsing؛ از یک سند خام تا داده‌ای که بتوان به آن اعتماد کرد"
description: "راهنمای تصمیم‌گیری و طراحی Pipeline پردازش اسناد؛ از انتخاب بین PDF Parser، OCR و VLM تا ساخت محصول، با تمرکز ویژه بر فارسی"
excerpt: "راهنمای تصمیم‌گیری و طراحی Pipeline پردازش اسناد؛ از انتخاب بین PDF Parser، OCR و VLM تا ساخت محصول، با تمرکز ویژه بر فارسی"
date: 2026-11-26 00:00:00 +0000
permalink: /fa/writing/document-parsing/
nav_exclude: true
image: /assets/img/social-card.png
feed:
  excerpt_only: true
tags: [document-parsing, OCR, VLM, PDF, RAG]
---
# مهندسی Document Parsing؛ از یک سند خام تا داده‌ای که بتوان به آن اعتماد کرد

**راهنمای تصمیم‌گیری و طراحی Pipeline پردازش اسناد؛ از انتخاب بین PDF Parser، OCR و VLM تا ساخت محصول، با تمرکز ویژه بر فارسی**

فرض کنید قرار است برای یک شرکت، دانشگاه یا سازمان یک سیستم RAG یا AI Agent بسازید. چند هزار صفحه PDF، فایل Word و سند اسکن‌شده در اختیارتان قرار گرفته و انتظار این است که سیستم بتواند بر اساس آن‌ها جواب بدهد. اولین کاری که احتمالاً به ذهنتان می‌رسد این است که متن فایل‌ها را استخراج کنید، Chunk بسازید و آن‌ها را به Vector Database بدهید.

اما اگر در همان مرحلهٔ استخراج، نرخ ۱۸ درصد به ۸۱ درصد تبدیل شود، پاورقی یک بند حقوقی به بند دیگری بچسبد، یا عدد کاملاً درست خوانده شود ولی در ستون اشتباه جدول قرار بگیرد، حتی یک مدل زبانی بسیار قدرتمند هم ممکن است با استناد به همان داده، پاسخ کاملاً اشتباهی تولید کند.

در پروژه‌هایی که روی پردازش اسناد فارسی داشتم، به‌خصوص اسناد مالی و سازمانی، بارها به همین مسئله برخوردم. بعضی از خطاها واقعاً مربوط به مدل OCR بودند؛ بعضی دیگر به نحوهٔ خواندن PDF، بازسازی جدول، جهت متن یا حتی Cache و ذخیره‌سازی خروجی مربوط می‌شدند. بعضی راهکارها در یک نمونه جواب می‌دادند و روی سند بعدی شکست می‌خوردند.

در این مطلب می‌خواهم آن تجربه‌ها را به یک مسیر آموزشی تبدیل کنم: اگر قرار است خودتان یک Document Parsing Pipeline طراحی کنید، از کجا شروع کنید؟ هر مرحله چه مسئولیتی دارد؟ از کجا بفهمید مشکل در کدام لایه است؟ چه ابزارهایی ارزش امتحان کردن دارند و چطور باید تصمیم بگیرید که نتیجه واقعاً بهتر شده است؟

این مقاله آموزش کدنویسی نیست؛ راهنمای طراحی Pipeline است. مثال‌های تجربی از مجموعه‌ای از اسناد مالی و سازمانی فارسی، شامل PDFهای دیجیتال و نواحی تصویری، گرفته شده‌اند. جزئیات محرمانهٔ پروژه و وضعیت داخلی توسعه در این مطلب موضوع بحث نیستند. برای خواندن شواهد از سه برچسب استفاده می‌کنم: **[آزمایش محدود]** یعنی مشاهدهٔ ثبت‌شده در نمونه‌های مشخص (نه مقایسهٔ هم‌شرایط همهٔ مدل‌ها)، **[مثال فرضی]** یعنی دادهٔ آموزشی، و **[مسیر پیشنهادی]** یعنی روشی که هنوز باید روی دادهٔ مقصد ارزیابی شود. نتایج منتشرشدهٔ دیگران با لینک منبع می‌آیند؛ گزارش‌های کامیونیتی تجربهٔ فردی‌اند.

### اگر فقط یک PDF و یک مدل قوی داشته باشیم، چرا این همه پیچیدگی؟

یک PDF کوتاه را می‌توان مستقیماً به VLM داد و جواب همان سؤال را بررسی کرد. برای محصولی که باید هزاران سند را به دادهٔ کامل، قابل‌تکرار و قابل‌اصلاح تبدیل کند، قرارداد خروجی فرق می‌کند. هیچ Framework مشخصی اجباری نیست؛ VLM می‌تواند موتور اصلی Parser باشد. در بخش ۳ مسئولیت‌ها و مسیر انتخاب را دقیق تفکیک می‌کنیم.

**موضع من دربارهٔ فارسی:** برای اسناد اسکن‌شده و عکس‌های فارسی، به‌خصوص اسناد مالی، سازمانی و ترکیبی، مسیر پیشنهادی من **VLM-first** است. حتی Bina که یکی از جدی‌ترین تلاش‌های تخصصی OCR فارسی است، در آزمایش محدود من به سطح اطمینان لازم نرسید. این یک تصمیم مهندسی برای کاربرد حساس است، نه حکم اثبات‌شده دربارهٔ تمام موتورهای OCR یا تمام نسخه‌های آینده. برای PDF دیجیتال سالم، Native-first همچنان Baseline مهمی است؛ اما وقتی محصول استخراج فیلدهای متنوع و فهم رابطه‌های بصری می‌خواهد، VLM مستقیم را حتی روی PDF دیجیتال هم به‌عنوان مسیر مستقل مقایسه می‌کنم.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-1-title" direction="ltr" height="503" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 503" width="720" xml:lang="fa"><title id="v5-figure-1-title">مسئولیت‌های خوانندهٔ فایل، OCR و Document Parser</title><rect x="0" y="0" width="720" height="503" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="23" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="28">یک سند، چند مسئولیت متفاوت</text><rect fill="#e1f1ef" height="55" rx="10" stroke="#176e73" width="480" x="120" y="49"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="82">PDF، تصویر یا فایل Office</text><path d="M360 104V126H180V145M360 126H540V145" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="92" rx="10" stroke="#aab8b5" width="330" x="15" y="145"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="180" y="173">خوانندهٔ فایل / PDF Parser</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="180" y="197">خواندن محتوای موجود</text><text direction="rtl" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="219">متن و هندسه</text><rect fill="#fffdf9" height="92" rx="10" stroke="#aab8b5" width="330" x="375" y="145"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="540" y="173">خوانش تصویر (وظیفهٔ OCR)</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="540" y="197">OCR کلاسیک، یادگیری‌محور یا VLM</text><text direction="rtl" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="219">خوانش‌های پیشنهادی تصویر</text><path d="M180 237V262H360M540 237V262H360V285" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#e1f1ef" height="80" rx="10" stroke="#176e73" width="520" x="100" y="285"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="313">بازسازی ساختار سند (Parsing)</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="336">چیدمان، ترتیب خواندن، جدول و روابط</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="355">یک موتور یا چند مؤلفه</text><path d="M360 365V391m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="64" rx="10" stroke="#aab8b5" width="440" x="140" y="391"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="416">اعتبارسنجی و حفظ شاهد اصلی</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="439">خروجی ساختاریافته، بازبینی یا رد</text></g></svg><figcaption>OCR یک وظیفه است، VLM خانواده‌ای از مدل‌هاست و Parser یک مسئولیت نرم‌افزاری دارد. یک مدل تخصصی یا عمومی ممکن است چند وظیفه را با هم انجام دهد؛ نمودار تقسیم منطقی مسئولیت‌هاست، نه ترتیب اجباری اجرا.</figcaption></figure>

### سؤال‌هایی که در کامیونیتی واقعاً مطرح می‌شوند

در بازخوانی گفت‌وگوهای گروه و پست‌های کانال، از اکتبر تا نوامبر ۲۰۲۶، بعد از کنار گذاشتن پیام‌های تکراری و بحث‌های نامرتبط، چند الگوی مهم به چشم می‌خورد. این یک تحلیل کیفی از نمونهٔ محدود گفت‌وگوهاست، نه نظرسنجی یا آمار بازار. در جدول زیر، سؤال‌ها را به زبان عمومی بازنویسی کرده‌ام و نام افراد را حذف کرده‌ام. ستون سوم برداشت مهندسی من از نیاز پشت سؤال است؛ قرار نیست بگوییم خود اعضا حتماً این مفاهیم را نمی‌دانستند.

| سؤال یا تجربهٔ مطرح‌شده | چیزی که باید از هم تفکیک کنیم | سؤال مهندسی پشت آن |
|---|---|---|
| «بهترین راه برای Parse کردن PDF فارسی برای چت چیست؟» (۱۱ اکتبر) | PDF تایپی، اسکن‌شده، ترکیبی و جدول‌دار | دامنهٔ پشتیبانی MVP چیست و ورودی نامعتبر چگونه رد می‌شود؟ |
| «می‌شود اسکن را فعلاً پشتیبانی نکنیم؟ OCR هزینه دارد.» (۱۱ اکتبر) | هزینهٔ قابلیت در مقابل هزینهٔ پذیرش ورودی ناقص | چه تعداد از مشتریان به اسکن نیاز دارند و حذف آن چه اثری دارد؟ |
| «Gemini یعنی PDF را به LLM بدهیم؟ چقدر هزینه می‌شود؟» (۱۱ اکتبر) | API با ورودی PDF، OCR، بازسازی سند و پاسخ به سؤال | سرویس چه چیزی تحویل می‌دهد، چه چیزی تکرار می‌شود و هزینهٔ کل چرخه چقدر است؟ |
| «فاکتورهای دست‌نویس با عکس‌های بی‌کیفیت را چطور بخوانیم؟» (۱۱ اکتبر) | فاکتور دیجیتال با قالب ثابت، اسکن، عکس و دست‌خط | چه فیلدهایی لازم‌اند، کیفیت تصویر چطور بررسی می‌شود و چه زمانی کاربر باید دوباره عکس بگیرد؟ |
| «قیمت‌ها خوب درمی‌آیند، ولی یک صفر کم می‌شود.» (۲۲ اکتبر) | دقت متن در برابر صحت عدد و تصمیم مالی | آیا می‌توان هر فیلد حساس را به شاهد اصلی وصل و پیش از ثبت کنترل کرد؟ |
| «مدل‌های Context طولانی دارند؛ چرا هنوز Chunking؟» (۱۸ نوامبر) | استخراج، Retrieval و استفادهٔ درست از Context | با حجم سند و تعداد پرس‌وجوی واقعی، کدام روش کیفیت و هزینهٔ بهتری دارد؟ |
| «PageIndex سریع‌تر و دقیق‌تر است؛ هزینه‌اش چطور؟» (۱۸ نوامبر) | زمان رسیدن به پاسخ صحیح در برابر Latency هر درخواست | هزینهٔ Ingest، جست‌وجو، Retry و بازبینی در برابر کیفیت مورد نیاز چقدر است؟ |
| «یک پروژهٔ OCR فارسی می‌خواهم، نه صرفاً یک مدل.» (۲۶ نوامبر) | موتور Recognition در برابر Pipeline شامل Layout، RTL، Normalization و Export | آیا محصول محلی، قابل نگهداری و با پوشش واقعی فارسی است یا فقط مدل تصویر به متن دارد؟ |

یک نکتهٔ جالب: در بحث فاکتور، پیشنهاد اولیهٔ خودم استفاده از Gemini و بهتر کردن کیفیت تصویر بود. یکی از اعضا به‌درستی به قالب ثابت فاکتور اشاره کرد. این سؤال باعث می‌شود توصیهٔ اولیه را دقیق‌تر کنیم: اگر سند دیجیتال، قالب ثابت و فیلدهای قابل‌استخراج دارد، اصلاً لازم نیست VLM را اجرا کنیم؛ اگر همان قالب روی عکس دست‌نویس آمده، هنوز باید پیکسل‌ها را بخوانیم. قالب ثابت، OCR را حذف نمی‌کند مگر اینکه متن معتبر دیجیتال در اختیار داشته باشیم. به همین شکل، پیشنهاد یک عضو برای PaddleOCR سبک‌تر و تجربهٔ دیگران با VLMهای چندوجهی، به‌خودی‌خود پاسخ «کدام بهترین است؟» نیست. باید روی همان نوع سند و همان فیلدها مقایسه کنیم.

این دغدغه‌ها محدود به گروه فارسی نیستند. در r/LocalLLaMA، یک نفر برای دیجیتالی کردن فرم‌های صنعتی و ورود به سیستم MES (Manufacturing Execution System؛ سامانهٔ مدیریت اجرای تولید)، با جدول‌های تو‌در‌تو و سخت‌افزار فقط CPU مشکل داشت؛ یک درخواست مستقیم Gemini روی جدول ساده خوب جواب داده بود، ولی روی ساختارهای پیچیده نه. نکتهٔ آموزشی این است که او به متن دقیق تمام دست‌نوشته‌ها نیاز نداشت؛ بخشی از مسئله کشف Schema فرم بود. در r/RAG، مهندسی با اسناد مالی محرمانهٔ ۱۵۰ تا ۲۰۰ صفحه‌ای، بین کیفیت یک سرویس ابری و محدودیت انتقال داده گیر کرده بود. در r/LangChain، گزارش‌های دیگری از مخلوط شدن ستون‌ها، خرابی جدول و ورود Header/Footer به محتوای اصلی وجود داشت. [۲۵](#ref-25) [۲۶](#ref-26) [۲۷](#ref-27)

در Hacker News نیز بحث PDF-to-text به سه مسئلهٔ مجزا کشیده شده بود: خواندن متن، استخراج فیلدهای مشخص و خودکارسازی گردش کار سازمانی. همچنین نیاز به دسترس‌پذیری و حفظ معنای ساختار برای خوانندهٔ انسانی مطرح شده بود. بعضی تجربه‌ها به نفع VLM مستقیم بودند و بعضی به نفع مسیرهای چندمرحله‌ای؛ این تفاوت بیشتر به نوع مسئله برمی‌گردد تا وجود یک مدل برنده برای همه. [۲۸](#ref-28)

**نتیجهٔ آموزشی:** به‌جای پرسیدن «بهترین OCR چیست؟» ابتدا بپرسید «ورودی چیست، خروجی دقیقاً چه چیزی باید باشد، خطای قابل تحمل چیست و محدودیت‌های محصول کدام‌اند؟» بحث‌های گروهی برای ساختن فرضیه و شناخت نیاز واقعی مفیدند، ولی جای Benchmark کنترل‌شده را نمی‌گیرند. در ادامهٔ مقاله از همین پرسش‌ها برای طراحی مسیر تصمیم استفاده می‌کنیم.

## ۱. قبل از انتخاب مدل، تعریف کنید Parsing موفق یعنی چه

**[مثال فرضی]** چنین جدولی را در یک سند مالی در نظر بگیرید.

| نوع تسهیلات | نرخ سود | سقف مبلغ |
|---|---:|---|
| خرید کالا | ۱۸٪ | ۵۰۰ میلیون ریال |
| سرمایه در گردش | ۲۳٪ | ۲ میلیارد ریال |
| قرض‌الحسنه | ۴٪ | ۳۰۰ میلیون ریال |

یک OCR ممکن است تمام کلمات و اعداد را بدون اشتباه بخواند. بااین‌حال اگر Parser نرخ ۲۳٪ را به قرض‌الحسنه نسبت دهد، اطلاعات خروجی غلط است. پس سؤال «متن درست خوانده شده؟» فقط یکی از سؤال‌های ماست.

برای طراحی Pipeline، اول مصرف‌کنندهٔ خروجی را مشخص کنید. جست‌وجو در یک کتاب متنی با استخراج فیلدهای قرارداد، تحلیل صورت مالی یا ساخت دستیار مستند یکسان نیست.

| کاربرد | حداقل چیزی که باید حفظ شود | خطای مهم |
|---|---|---|
| جست‌وجوی متن کتاب یا راهنما | پاراگراف، ترتیب خواندن و صفحه | مخلوط شدن دو ستون |
| استخراج داده از فاکتور و فرم | نام فیلد، مقدار، واحد و ارتباط آن‌ها | مبلغ درست با برچسب اشتباه |
| تحلیل جدول و گزارش مالی | سطر، ستون، سرستون، Span و پاورقی | جابه‌جایی معنای سلول |
| دستیار حقوقی یا سازمانی | سلسله‌مراتب بندها، تبصره، نسخه و شواهد | نسبت دادن شرط به بند دیگری |
| RAG با استناد | متن ساختاریافته، شناسه و منشأ هر عنصر | جواب مستند به دادهٔ خراب |

**Document Parsing یعنی بازسازی یک نمایش قابل‌استفاده از محتوای سند، ساختار آن و ارتباط اجزایش.** تبدیل PDF به یک رشتهٔ متن ممکن است برای بعضی کاربردها کافی باشد؛ برای خیلی از کاربردهای دیگر نیست.

یک نکتهٔ معماری: Parsing را از Information Extraction و پاسخ‌گویی جدا کنید. Parser باید تا جای ممکن چیزی را که در سند هست بازسازی کند. مرحلهٔ بعدی می‌تواند مثلاً مبلغ قرارداد را استخراج کند، محاسبه انجام دهد یا پاسخ کاربر را بسازد. اگر از همان ابتدا از مدل بخواهید سند را تفسیر و خلاصه کند، بخشی از شواهد ممکن است قبل از ورود به سیستم از دست بروند.

### پنج محصول متفاوت که معمولاً همه را «OCR» صدا می‌زنیم

| محصول | خروجی قابل تحویل | حداقل معیار پذیرش |
|---|---|---|
| چت یک‌باره با سند | پاسخ به سؤال و اشاره به شاهد مربوط | صحت همان پاسخ؛ نه ادعای استخراج کامل فایل |
| OCR و آرشیو قابل جست‌وجو | متن یا PDF دارای لایهٔ متن، همراه اصل | پوشش صفحات، خوانایی متن و خطاهای نویسه |
| تبدیل سند برای RAG | عناصر ساختاریافته، ترتیب، صفحه و روابط | Recall شاهد، کیفیت جدول و پوشش؛ قابلیت بازسازی و Re-index |
| پردازش فاکتور و فرم | رکورد فیلدها، مقدار خام، واحد، شاهد و وضعیت | Exact Match فیلد و ردیف، نرخ ثبت اشتباه، نرخ بازبینی |
| پلتفرم Document AI چندمشتری | خروجی نسخه‌دار برای چند کاربرد، Job و API | کیفیت به تفکیک نوع سند، p95 (صدک ۹۵ تأخیر)، Throughput، امنیت و هزینهٔ رکورد پذیرفته‌شده |

قابلیت‌هایی مانند استخراج فیلد، جدول، موقعیت منبع و بازبینی در Amazon Textract و Azure Document Intelligence نمونه‌هایی از قرارداد محصول‌اند. [۲۹](#ref-29)

برای شروع یک پروژه با کارفرما، قبل از انتخاب ابزار چند چیز را مشخص کنید: تعداد سند و صفحه در روز، درصد اسکن و دست‌نویس، زبان و قالب، فیلدهایی که نباید اشتباه شوند، نیاز به استناد یا خروجی برای ماشین، زمان قابل‌قبول پاسخ، بودجه و امکان ارسال داده به خارج از زیرساخت. اگر پاسخ این سؤال‌ها مشخص نیست، انتخاب مدل پیش از شناخت مسئله است. ممکن است محصول فعلاً فقط PDF دیجیتال سالم را بپذیرد، اما باید این محدودیت را صریح اعلام کند.

## ۲. سند را بشناسید؛ همهٔ PDFها مثل هم نیستند

PDF اساساً برای نمایش سند طراحی شده است. چیزی که در صفحه به‌عنوان عنوان، جدول یا یک پاراگراف می‌بینیم، ممکن است در فایل مجموعه‌ای از دستورات ترسیم متن، مختصات، خطوط و تصاویر باشد. البته بعضی PDFها ساختار منطقی و Tag هم دارند، اما نمی‌توان فرض کرد همهٔ فایل‌ها این اطلاعات را به‌درستی حفظ کرده‌اند. توضیحات [۱](#ref-1) و [۲](#ref-2) دربارهٔ محدودیت استخراج متن و ترتیب عناصر را ببینید.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-2-title" direction="ltr" height="313" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 313" width="720" xml:lang="fa"><title id="v5-figure-2-title">مقایسه نمای سند و نواحی هندسی استخراج‌شده؛ تصویر آموزشی شماتیک.</title><defs><marker id="arrow-2" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="313" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="230" rx="8" stroke="#aab8b5" width="190" x="95" y="12"/><rect fill="#5f6964" height="9" rx="2" stroke="none" width="93.1" x="171.0" y="30"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="110" y="54"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="110" y="66"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="110" y="78"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="110" y="90"/><rect fill="#fffdf9" height="70" rx="2" stroke="#aab8b5" width="160" x="110" y="112"/><rect fill="#e1f1ef" height="17.5" rx="1" stroke="none" width="160" x="110" y="112"/><path d="M110 129.5H270" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M110 147.0H270" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M110 164.5H270" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M163.33333333333334 112V182" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M216.66666666666669 112V182" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="110" y="194"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="148" x="122" y="204"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="136" x="134" y="214"/><rect fill="#fffdf9" height="230" rx="8" stroke="#aab8b5" width="190" x="435" y="12"/><rect fill="#5f6964" height="9" rx="2" stroke="none" width="93.1" x="511.0" y="30"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="450" y="54"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="450" y="66"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="450" y="78"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="450" y="90"/><rect fill="#fffdf9" height="70" rx="2" stroke="#aab8b5" width="160" x="450" y="112"/><rect fill="#e1f1ef" height="17.5" rx="1" stroke="none" width="160" x="450" y="112"/><path d="M450 129.5H610" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M450 147.0H610" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M450 164.5H610" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M503.3333333333333 112V182" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M556.6666666666666 112V182" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="450" y="194"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="148" x="462" y="204"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="136" x="474" y="214"/><rect fill="none" height="20" rx="3" stroke="#176e73" stroke-dasharray="5 3" stroke-width="2" width="104.50000000000001" x="505.3" y="24"/><rect fill="none" height="55" rx="3" stroke="#176e73" stroke-dasharray="5 3" stroke-width="2" width="170" x="445" y="47"/><rect fill="none" height="80" rx="3" stroke="#176e73" stroke-dasharray="5 3" stroke-width="2" width="170" x="445" y="107"/><rect fill="none" height="35" rx="3" stroke="#176e73" stroke-dasharray="5 3" stroke-width="2" width="170" x="445" y="190"/><path d="M300 120L420 120" fill="none" marker-end="url(#arrow-2)" stroke="#176e73" stroke-width="2"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="190" y="267">چیزی که انسان می‌بیند</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="530" y="267">آنچه Parser تحلیل می‌کند</text></g></svg><figcaption>مقایسه نمای سند و نواحی هندسی استخراج‌شده؛ [مثال شماتیک]</figcaption></figure>

| نوع ورودی | چه چیزی در اختیار داریم؟ | مسیر اولیهٔ پیشنهادی |
|---|---|---|
| Born-digital PDF | متن دیجیتال، مختصات و اجزای گرافیکی | Native Extraction؛ سپس بررسی Layout |
| Scanned PDF | تصویر صفحات، گاهی با لایهٔ OCR قدیمی | تحلیل کیفیت تصویر و خوانش بصری؛ برای فارسی حساس، VLM-first |
| Hybrid PDF | ترکیب متن دیجیتال، تصویر و جدول اسکن‌شده | تصمیم‌گیری در سطح صفحه یا ناحیه |
| DOCX | پاراگراف، Style، جدول، فهرست و رابطه‌های داخل فایل | خواندن ساختار اصلی Word؛ رندر فقط در صورت نیاز |
| XLSX | سلول، Sheet، محدودهٔ ادغام‌شده و فرمول | خواندن مدل Spreadsheet؛ تفکیک فرمول و مقدار نمایش‌داده‌شده |
| تصویر | پیکسل؛ بدون لایهٔ متن | پیش‌پردازش هدفمند، Detection و Recognition؛ برای فارسی حساس، VLM-first |

فرض کنید PDF صدصفحه‌ای شما ۹۵ صفحهٔ دیجیتال، سه صفحهٔ اسکن‌شده و دو صفحهٔ دارای جدول تصویری دارد. اجرای OCR روی کل فایل احتمالاً اولین انتخاب خوبی نیست؛ از طرف دیگر استخراج مستقیم متن هم دو جدول تصویری را نمی‌خواند.

یک اشتباه رایج این است که فقط وجود متن را بررسی کنیم و بگوییم صفحه دیجیتال است. ممکن است صفحه یک عنوان دیجیتال و یک جدول تصویری داشته باشد. حتی ممکن است متن قابل انتخاب باشد، ولی از OCR قدیمی یا نگاشت خراب فونت آمده باشد.

در مرحلهٔ شناخت ورودی، برای هر صفحه به این سؤال‌ها جواب بدهید: آیا متن قابل استخراج است؟ آیا با ظاهر صفحه تطابق دارد؟ تصویر، جدول یا فرمول جداگانه وجود دارد؟ سند چرخیده یا کج است؟ اطلاعات حساس کجا قرار دارند؟ این ارزیابی اولیه را ذخیره کنید؛ بعداً مبنای Routing و Debugging می‌شود.

**PDF ظاهراً دیجیتال (Fake Digital):** صفحهٔ اسکن‌شده ممکن است تصویر تمام‌صفحه و یک لایهٔ OCR نامرئی داشته باشد. به‌عنوان Heuristic پیشنهادی، اگر یک تصویر یا اجتماع نواحی تصویری بیش از حدود ۸۰ تا ۹۰ درصد مساحت قابل‌مشاهدهٔ صفحه را پوشانده و بخش عمدهٔ متن در حالت نمایش نامرئی PDF (Text Rendering Mode 3) قرار دارد، صفحه را کاندیدای «اسکن با لایهٔ متن نیازمند ممیزی» بدانید. در PyMuPDF از اطلاعات تصویر و BBox، و برای متن از Trace/نوع نمایش (type 3 در get_texttrace، مطابق نسخهٔ API) استفاده کنید؛ در سطح PDF نیز عملگر `Tr` و وضعیت گرافیکی را بررسی کنید. مساحت اشتراک و برش‌ها را حساب کنید، نه جمع سادهٔ کادرهای هم‌پوشان. این آستانه قراردادی است، نه استاندارد یا دلیل قطعی خرابی: سند ممکن است OCR عالی یا تصویر پس‌زمینه داشته باشد؛ تصاویر تکه‌تکه، شفافیت و لایه‌های پوشاننده هم حالت‌های دیگرند. چند ناحیهٔ متن و فیلد حساس را با تصویر مقایسه کنید؛ اگر لایهٔ موجود معتبر است دوباره OCR نکنید، اگر بخشی خراب است فقط همان بخش را بازیابی کنید. [۲](#ref-2) [۱۵](#ref-15)

برای فرمت‌های ساختاریافته، خودتان را مجبور نکنید همهٔ فایل‌ها را به PDF تبدیل کنید. مثلاً در Word می‌توانید از Style عنوان و ساختار جدول استفاده کنید؛ اگر ابتدا آن را چاپ مجازی کنید، ممکن است بخشی از این روابط را از دست بدهید. در PDF هم همیشه شمارهٔ فیزیکی صفحه با شماره‌ای که پایین صفحه چاپ شده یکی نیست.

## ۳. نقشهٔ ذهنی Pipeline؛ فرق OCR، Layout، Reading Order و Table Recognition

برای پیدا کردن ریشهٔ خطا، مسئولیت هر لایه را جدا تعریف کنید.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-3-title" direction="ltr" height="393" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 393" width="720" xml:lang="fa"><title id="v5-figure-3-title">چهار مسئولیت متفاوت در پردازش سند.</title><rect x="0" y="0" width="720" height="393" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#f6f5f0" height="165" rx="8" stroke="#aab8b5" width="335" x="15" y="10"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="182" y="37">OCR</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="182" y="160">خواندن متن</text><rect fill="#fffdf9" height="76" rx="8" stroke="#aab8b5" width="205" x="80" y="55"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="26" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="182" y="97">نرخ ۱۸٪</text><rect fill="#f6f5f0" height="165" rx="8" stroke="#aab8b5" width="335" x="370" y="10"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="37">تحلیل چیدمان</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="160">تشخیص ناحیه</text><rect fill="#fffdf9" height="85" rx="8" stroke="#aab8b5" width="110" x="482" y="52"/><rect fill="#e1f1ef" height="15" rx="2" stroke="#176e73" width="88" x="493" y="61"/><rect fill="#e1f1ef" height="43" rx="2" stroke="#176e73" width="88" x="493" y="81"/><rect fill="#f6f5f0" height="165" rx="8" stroke="#aab8b5" width="335" x="15" y="190"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="182" y="217">ساختار جدول</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="182" y="340">بازسازی سلول</text><rect fill="#fffdf9" height="82" rx="2" stroke="#aab8b5" width="205" x="80" y="233"/><rect fill="#e1f1ef" height="20.5" rx="1" stroke="none" width="205" x="80" y="233"/><path d="M80 253.5H285" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M148.33333333333331 274.0H285" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M148.33333333333331 294.5H285" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M148.33333333333331 233V315" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M216.66666666666666 233V315" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#f6f5f0" height="165" rx="8" stroke="#aab8b5" width="335" x="370" y="190"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="217">بازسازی سند</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="340">ساختار نهایی</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="251">DocumentIR</text><path d="M537 258V274H459V284 M537 274H616V284" fill="none" stroke="#aab8b5" stroke-width="2"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="459" y="304">متن</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="616" y="304">TableIR</text></g></svg><figcaption>چهار مسئولیت متفاوت در پردازش سند.</figcaption></figure>

| مؤلفه | ورودی و خروجی | سؤال اصلی |
|---|---|---|
| PDF Decoder / Native Extraction | فایل PDF ← نویسه‌ها، خطوط، تصاویر و مختصات | داخل فایل چه اطلاعاتی ذخیره شده است؟ |
| Image Preprocessing | تصویر ← نسخهٔ اصلاح‌شده و تبدیل هندسی | آیا تصویر برای خواندن آماده است؟ |
| Text Detection | تصویر ← کادر یا چندضلعی متن | متن در کجای تصویر است؟ |
| Text Recognition / OCR | ناحیهٔ متن ← رشتهٔ نویسه‌ها | چه چیزی نوشته شده است؟ |
| Layout Analysis | صفحه ← ناحیه‌های عنوان، متن، جدول، شکل و غیره | نوع و محل هر عنصر چیست؟ |
| Reading Order | عناصر ← ترتیب خواندن | از کدام بخش به کدام بخش برویم؟ |
| Table Structure Recognition (TSR) | جدول ← سطر، ستون، سلول و Span | هر مقدار متعلق به کدام جایگاه است؟ |
| Formula / Chart Recognition | ناحیهٔ فرمول یا نمودار ← نمایش ساختاری یا خوانش مستند | رابطهٔ نمادها یا داده‌ها چیست؟ |
| Reconciliation | چند خوانش و هندسه ← کاندیداهای تطبیق‌یافته و تعارض‌ها | کدام اطلاعات با هم سازگارند؟ |
| Structural Reconstruction | اجزا و روابط ← سند ساختاریافته | همهٔ این اطلاعات چگونه به هم وصل می‌شوند؟ |
| Validation | خروجی و شواهد ← وضعیت و دلایل پذیرش یا رد | چه چیزی واقعاً قابل انتشار است؟ |

این‌ها مسئولیت منطقی‌اند، نه الزاماً ده سرویس مستقل یا یک ترتیب اجرایی ثابت. یک مدل یا Parser ممکن است چند وظیفه را با هم انجام دهد. در بعضی Pipelineها Layout قبل از OCR اجرا می‌شود تا ناحیه‌ها جدا شوند؛ در بعضی دیگر OCR و Layout از اطلاعات یکدیگر استفاده می‌کنند. مدل End-to-End نیز ممکن است همه را در یک پاس تولید کند. مهم این است که حتی در حالت End-to-End بتوانید خطای متن، ساختار و ترتیب را جداگانه بسنجید. [۳](#ref-3) [۱۰](#ref-10)

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-4-title" direction="ltr" height="618" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 618" width="720" xml:lang="fa"><title id="v5-figure-4-title">نقشهٔ مفهومی چند مسیر پردازش: Provider، Docling محلی یا بازیابی تصویری. این‌ها گزینه‌اند، نه ترتیب اجباری یا معماری تأییدشدهٔ عمومی. فقط دادهٔ واجد شرایط به مرحلهٔ بعد می‌رود.</title><defs><marker id="arrow-4" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="618" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#f6f5f0" height="44" rx="8" stroke="#aab8b5" width="470" x="125" y="12"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="40">PDF و شناسه / هش منبع</text><path d="M360 56L360 74" fill="none" marker-end="url(#arrow-4)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="44" rx="8" stroke="#aab8b5" width="470" x="125" y="74"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="102">PyMuPDF؛ متن و هندسهٔ صفحه</text><path d="M360 118L360 136" fill="none" marker-end="url(#arrow-4)" stroke="#176e73" stroke-width="2"/><path d="M360 136H120V151 M360 136V151 M360 136H600V151" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="62" rx="8" stroke="#176e73" width="220" x="10" y="151"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="120" y="176">سرویس پردازش</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="120" y="197">+ Docling / TableFormer</text><rect fill="#e1f1ef" height="62" rx="8" stroke="#176e73" width="220" x="250" y="151"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="176">Docling محلی</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="197">TableFormer</text><rect fill="#e1f1ef" height="62" rx="8" stroke="#176e73" width="220" x="490" y="151"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="600" y="176">مسیر محلی و VLM</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="600" y="197">برش‌های منتخب</text><path d="M120 213V230H360 M360 213V245 M600 213V230H360" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="44" rx="8" stroke="#aab8b5" width="470" x="125" y="245"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="273">بازسازی ساختار</text><path d="M360 289L360 307" fill="none" marker-end="url(#arrow-4)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="44" rx="8" stroke="#aab8b5" width="470" x="125" y="307"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="335">DocumentIR / TableIR</text><path d="M360 351L360 369" fill="none" marker-end="url(#arrow-4)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="44" rx="8" stroke="#aab8b5" width="470" x="125" y="369"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="397">کنترل‌های پذیرش</text><path d="M360 413L360 431" fill="none" marker-end="url(#arrow-4)" stroke="#176e73" stroke-width="2"/><path d="M360 431H180V448 M360 431H540V448" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#e7f4ed" height="48" rx="8" stroke="#15704e" width="320" x="20" y="448"/><text direction="rtl" fill="#15704e" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="478">محتوای پذیرفته‌شده</text><rect fill="#fff1d9" height="48" rx="8" stroke="#94600d" width="320" x="380" y="448"/><text direction="rtl" fill="#94600d" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="478">بازبینی / قرنطینه</text><path d="M180 496L180 520" fill="none" marker-end="url(#arrow-4)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="56" rx="8" stroke="#aab8b5" width="320" x="20" y="520"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="543">JSON / HTML / Markdown</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="562">قطعه‌های قابل انتشار</text></g></svg><figcaption>نقشهٔ مفهومی چند مسیر پردازش: Provider، Docling محلی یا بازیابی تصویری. این‌ها گزینه‌اند، نه ترتیب اجباری یا معماری تأییدشدهٔ عمومی. فقط دادهٔ واجد شرایط به مرحلهٔ بعد می‌رود.</figcaption></figure>

### سه اصطلاح، سه سطح تصمیم: Parser، OCR و VLM

| اصطلاح | دقیقاً چیست؟ | خروجی معمول | چه چیزی را تضمین نمی‌کند؟ |
|---|---|---|---|
| PDF Parser / Decoder | نرم‌افزاری که ساختار فایل PDF و اطلاعات موجود در آن را می‌خواند | متن دیجیتال، مختصات، اجزای گرافیکی و رندر صفحه | صحت نگاشت Unicode یا وجود رابطهٔ منطقی جدول |
| OCR کلاسیک یا یادگیری‌محور | وظیفهٔ تشخیص و خواندن متن از تصویر؛ ممکن است Detection و Recognition جدا یا یکپارچه باشند | متن، گاهی کادر کلمات و Confidence | صحت ساختار سطر و ستون، معنای متن و تضمین درستی مقدار |
| مدل Layout | مدلی برای یافتن و دسته‌بندی عناصر صفحه | کادر عنوان، متن، جدول، شکل و غیره | خواندن درست محتوای هر کادر یا ترتیب کامل سند |
| VLM تخصصی سند | مدل آموزش‌دیده برای خوانش یا بازسازی سند | متن، Markdown، HTML، DocTags یا کادرها، بسته به مدل | کامل بودن استخراج و دقت زبان یا حوزهٔ کاربرد شما |
| VLM عمومی | مدل چندوجهی با توانایی پردازش تصویر و زبان | پاسخ، تفسیر، رونویسی یا دادهٔ ساختاریافته | صحت معنایی، عدم Hallucination، پوشش همهٔ عناصر و مختصات معتبر |
| Document Parser | ترکیب یک یا چند روش برای تبدیل سند به نمایش قابل‌استفاده | سند ساختاریافته، روابط، شواهد، وضعیت‌ها | کیفیت Production بدون اعتبارسنجی و تست روی دادهٔ مقصد |

Tesseract و VLM هر دو می‌توانند کار OCR انجام دهند، اما قرارداد خروجی‌شان فرق دارد. موتور OCR متعارف معمولاً با تنظیم زبان، قطعه‌بندی و گزینه‌های Recognition هدایت می‌شود؛ مدل چندوجهی دستورپذیر را می‌توان با Prompt زبان طبیعی برای یک وظیفهٔ مشخص هدایت کرد. این تفاوت دربارهٔ موتور Recognition است، نه همهٔ سرویس‌های تجاری OCR که ممکن است خودشان Schema، استخراج فیلد یا VLM داشته باشند. توضیح عملی این مزیت در بخش ۱۰ آمده است. Docling چارچوب پردازش است، Granite-Docling مدل تخصصی، MarkItDown مبدل فرمت، و LlamaParse و Document AI سرویس‌اند.

### چرا اصلاً Parser داشته باشیم اگر VLM می‌تواند کل PDF را بخواند؟

اگر منظورتان از Parser یک Framework اضافی است، پاسخ این است که **همیشه لازم نیست**. یک مدل مستقیم می‌تواند بهترین Baseline برای یک سند کوچک و یک سؤال محدود باشد. اگر منظورتان یک مسیر قابل اعتماد از فایل تا داده است، حتی وقتی تمام Recognition و Layout را به VLM واگذار می‌کنید، باز هم به بخشی برای دریافت فایل، قرارداد خروجی، کنترل پوشش، اعتبارسنجی و نگهداری شاهد نیاز دارید. در آن حالت VLM موتور Parser است، نه جایگزین تمام نیازهای محصول.

چند مثال برای فهم تفاوت:

- وقتی یک گزارش دیجیتال سالم است، متن واقعی و مختصات از PDF قابل خواندن‌اند. تبدیل همهٔ صفحات به تصویر و بازتولید متن توسط مدل ممکن است یک نسخهٔ سالم را با یک تخمین جایگزین کند. در این مورد مستقیم خواندن فایل مسیر پایهٔ مناسبی است.
- وقتی فقط می‌خواهید بدانید یک فاکتور متعلق به کدام شرکت است، مدل بصری مستقیم می‌تواند مفید باشد. اما برای ثبت خودکار همهٔ اقلام فاکتور، باید کامل بودن ردیف‌ها و ارتباط مبلغ و واحد را بررسی کنید.
- وقتی می‌خواهید از یک نمودار بدون برچسب عددی نتیجه بگیرید، OCR به‌تنهایی کافی نیست. VLM یا تحلیل تخصصی نمودار می‌تواند معنای بصری را بهتر پوشش دهد؛ مقدار تخمینی را نباید به‌عنوان عدد عیناً نوشته‌شده ثبت کنید.
- وقتی یک سند برای کاربران زیادی دوباره استفاده می‌شود، استخراج نسخه‌دار و قابل اصلاح می‌تواند هزینهٔ تکرار Parsing و زمان دیباگ را کم کند. در مقابل اگر یک بار پردازش می‌کنیم، ممکن است هزینهٔ مهندسی یک Pipeline پیچیده بیشتر از استفادهٔ مستقیم از مدل و بازبینی انسانی باشد.

آیا فرستادن PDF به API به معنی حذف Parsing است؟ نه لزوماً. طبق مستندات، APIهای دارای قابلیت PDF در OpenAI متن استخراج‌شده و تصویر صفحات را به مدل می‌دهند؛ Claude نیز از متن و تصاویر صفحات استفاده می‌کند و Gemini قابلیت درک مستقیم سند را ارائه می‌دهد. سرویس بخشی از آماده‌سازی ورودی را از شما پنهان می‌کند. یک پاسخ صحیح به سؤال، اثبات نمی‌کند تمام صفحات، تمام سلول‌ها یا تمام مختصات به شکل قابل‌استفاده به شما برگشته‌اند. همچنین نوع مدل، Endpoint و روش ارسال فایل مهم‌اند؛ رفتار رابط Chat را به API یا مدل محلی تعمیم ندهید. [۳۰](#ref-30)

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-17-title" direction="ltr" height="508" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 508" width="720" xml:lang="fa"><title id="v5-figure-17-title">تفاوت پردازش هنگام پرس‌وجو و هنگام ورود سند</title><rect x="0" y="0" width="720" height="508" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="22" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="27">پرسش یک‌باره یا دادهٔ قابل استفادهٔ مجدد</text><rect fill="#e1f1ef" height="48" rx="10" stroke="#176e73" width="330" x="15" y="52"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="180" y="83">خوانش هنگام پرسش</text><rect fill="#e1f1ef" height="48" rx="10" stroke="#176e73" width="330" x="375" y="52"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="540" y="83">پردازش هنگام ورود سند</text><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="15" y="128"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="162">سند و پرسش</text><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="375" y="128"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="162">سند اصلی</text><path d="M180 185V211m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><path d="M540 185V211m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="15" y="211"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="245">متن یا VLM مستقیم</text><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="375" y="211"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="245">بازسازی و اعتبارسنجی</text><path d="M180 268V294m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><path d="M540 268V294m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="15" y="294"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="328">تطبیق پاسخ با منبع</text><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="375" y="294"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="328">شواهد ساختاریافته و نسخه‌دار</text><path d="M180 351V377m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><path d="M540 351V377m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="15" y="377"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="411">پاسخ همین پرسش</text><rect fill="#fffdf9" height="57" rx="10" stroke="#aab8b5" width="330" x="375" y="377"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="411">جست‌وجو، پاسخ، خروجی یا بازبینی</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="180" y="460">سنجش صحت پاسخ</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="540" y="460">سنجش کامل بودن و استفادهٔ مجدد</text></g></svg><figcaption>دو قرارداد متفاوت: جواب به یک سؤال در لحظه، یا ساخت دادهٔ قابل استفادهٔ مجدد. هیچ‌کدام ذاتاً برنده نیستند؛ ارزیابی، هزینه، نیاز به شواهد و دفعات استفاده تعیین می‌کند کدام مناسب است.</figcaption></figure>

### نقشهٔ تصمیم‌گیری سریع؛ اول مسئله را انتخاب کنید، بعد ابزار را

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-5-title" direction="ltr" height="714" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 714" width="720" xml:lang="fa"><title id="v5-figure-5-title">انتخاب مسیر بر اساس محصول و کیفیت محتوای دیجیتال</title><rect x="0" y="0" width="720" height="714" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="24" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="28">ساده‌ترین مسیر کافی را انتخاب کنید</text><rect fill="#e1f1ef" height="54" rx="10" stroke="#176e73" width="480" x="120" y="48"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="80">محصول چه خروجی می‌خواهد؟</text><path d="M360 102V127H180V148M360 127H540V148" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="80" rx="10" stroke="#aab8b5" width="330" x="15" y="148"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="180" y="178">یک پرسش محدود؟</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="180" y="202">متن یا VLM مستقیم + بررسی شاهد</text><rect fill="#fffdf9" height="80" rx="10" stroke="#aab8b5" width="330" x="375" y="148"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="540" y="178">استخراج قابل استفادهٔ مجدد؟</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="540" y="202">بررسی فایل، صفحه و ناحیه</text><path d="M540 228V248H360V273" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#e1f1ef" height="54" rx="10" stroke="#176e73" width="480" x="120" y="273"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="305">آیا متن دیجیتال قابل‌اعتماد داریم؟</text><path d="M360 327V350H180V371M360 350H540V371" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="13" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="180" y="344">بله</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="13" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="540" y="344">خیر / ناقص</text><rect fill="#fffdf9" height="80" rx="10" stroke="#aab8b5" width="330" x="15" y="371"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="180" y="401">خوانندهٔ Native / فرمت‌محور</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="180" y="425">حفظ متن و هندسهٔ اصلی</text><rect fill="#fffdf9" height="80" rx="10" stroke="#aab8b5" width="330" x="375" y="371"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="540" y="401">خوانش فارسی: VLM-first</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="540" y="425">OCR تخصصی برای مقایسه</text><path d="M180 451V475H360M540 451V475H360V494" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#e1f1ef" height="78" rx="10" stroke="#176e73" width="570" x="75" y="494"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="524">بازسازی ساختار در صورت نیاز</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="549">کنترل پوشش، فیلدها و شاهد؛ بازبینی ابهام</text></g><rect x="25" y="624" width="670" height="75" rx="9" fill="#e1f1ef" stroke="#176e73"/><text x="360" y="648" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" direction="rtl" unicode-bidi="isolate">دروازهٔ پذیرش: پوشش و پذیرش غلط</text><text x="360" y="676" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="400" text-anchor="middle" direction="rtl" unicode-bidi="isolate">نرخ بازبینی، قرنطینه و هزینهٔ هر رکورد پذیرفته‌شده</text></svg><figcaption>مسیر تصمیم پیشنهادی: پرسش محدود ممکن است نیازی به استخراج ماندگار نداشته باشد. برای استخراج قابل استفادهٔ مجدد، محتوای دیجیتال معتبر را حفظ کنید و خوانش تصویری را در صورت نیاز اضافه کنید. ساختار و کیفیت را مستقل از نوع موتور بسنجید. مسیر پیشنهادی برای فارسی حساس: VLM-first برای خوانش بصری؛ OCR تخصصی برای مقایسه یا کاربرد تأییدشده.</figcaption></figure>

| وضعیت واقعی سند | Baseline پیشنهادی | چه زمانی مسیر را تغییر دهیم؟ |
|---|---|---|
| PDF دیجیتال، متن سالم، یک ستون | Native Extraction | اگر Heading، جدول یا ترتیب از دست می‌رود، Layout و Reconstruction اضافه کنید |
| PDF دیجیتال با جدول پیچیده | Native + ساختار جدول / Layout | اگر متن سلول خراب است، OCR یا VLM همان ناحیه؛ اگر هندسه خراب است، مدل ساختار |
| اسکن فارسی / تصویر فارسی | VLM-first برای کیفیت بالا؛ مدل تخصصی OCR به‌عنوان Baseline مقایسه | اگر خوانش، پوشش یا فیلد حساس درست نیست، Crop، مدل جایگزین یا بازبینی |
| اسکن واضح در زبان‌های دیگر یا کاربرد کم‌ریسک | OCR تخصصی یا VLM بر اساس ارزیابی | اگر علائم و ساختار مهم خراب‌اند، ناحیه‌ای مقایسه و ارزیابی کنید |
| صفحهٔ ترکیبی | حفظ Native سالم + خوانش تصویری بخش لازم | اگر Router ناحیه را جا انداخت، Coverage و تشخیص تصویر را اصلاح کنید |
| عکس دست‌نویس یا فرم نامنظم | برای فارسی VLM-first؛ OCR/HTR تخصصی در مقایسهٔ مستقل | اگر قابل خواندن نیست، Review یا درخواست عکس بهتر؛ نه حدس مدل |
| DOCX یا Excel | خوانندهٔ ساختار اصلی فرمت | VLM فقط برای محتوای تصویری یا تفسیر هدفمند در صورت نیاز |
| سؤال یک‌باره دربارهٔ نمودار یا سند کوچک | VLM مستقیم یا متن کامل + تصویر | اگر شاهد دقیق یا استخراج جامع لازم است، مسیر قابل‌ردیابی بسازید |

اگر بیشتر صفحات ساده‌اند، Routing را آزمایش کنید؛ اگر تقریباً همه پیچیده‌اند، مسیر End-to-End شاید ساده‌تر باشد. **متن دیجیتال سالم را با رونویسی OCR جایگزین نکنید.** فایل اصلی و لایهٔ Native را حفظ کنید؛ حتی اگر برای Layout صفحه را رندر می‌کنید. در صفحهٔ ترکیبی، فقط ناحیهٔ مشکوک را به OCR یا VLM بفرستید و در صورت نیاز شاهد سرستون و واحد را هم همراهش ارسال کنید. اگر Crop زمینه را از بین می‌برد، مسیر تمام‌صفحه را نیز مقایسه کنید.

VLM-first یعنی موتور اصلی خوانش پیکسل‌های فارسی را از میان VLMهای مناسب انتخاب کنیم، نه اینکه الزاماً هر صفحه را کامل به بزرگ‌ترین مدل ابری بفرستیم. OCR تخصصی را فقط در مسیرهایی وارد پذیرش خودکار کنید که کیفیت آن روی دادهٔ مقصد ثابت شده است. در اسناد دیجیتال هم VLM مستقیم با ورودی PDF می‌تواند مسیر ساده‌تری برای استخراج هدفمند باشد؛ حفظ اصل PDF و متن Native برای شاهد و مقایسه همچنان مفید است.

دروازهٔ انتخاب مسیر فقط CER یا زمان اجرا نیست: نرخ بازبینی (Review Rate)، نرخ قرنطینه (Quarantine Rate)، نرخ پذیرش غلط، پوشش و **هزینهٔ هر رکورد پذیرفته‌شده** را کنار هم قرار دهید. Router ارزان که موارد سخت را پنهان می‌کند موفق نیست؛ Router محتاط که همهٔ خروجی‌ها را به اپراتور می‌فرستد نیز ممکن است هدف اقتصادی محصول را برآورده نکند. معیارها و مخرج آن‌ها در بخش ۱۳ تعریف شده‌اند.

## ۴. استخراج Native؛ چرا PDF خوانا لزوماً متن سالم ندارد؟

### PyMuPDF دقیقاً به چه درد می‌خورد؟

PyMuPDF یکی از کتابخانه‌هایی بود که برای دسترسی مستقیم به شواهد PDF از آن استفاده کردم. می‌توانیم متن دیجیتال، مختصات کلمات و نویسه‌ها، تصویر صفحه و اطلاعات هندسی را از آن بگیریم. این ابزار برای ساخت Baseline سریع و برای پیدا کردن محل دقیق خطاها ارزشمند است. pypdf و pdfplumber نیز گزینه‌هایی برای مقایسهٔ خواندن PDF هستند، نه الزاماً جایگزین کامل یک Parser آگاه از Layout. [۱](#ref-1) [۲](#ref-2) [۱۳](#ref-13)

Bounding Box یا BBox یعنی محدودهٔ هندسی یک عنصر. مثلاً اگر یک سلول جدول استخراج شده است، مختصات به ما امکان می‌دهد همان ناحیه را در تصویر صفحه ببینیم و صحت متن یا مرز آن را بررسی کنیم.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-6-title" direction="ltr" height="280" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 280" width="720" xml:lang="fa"><title id="v5-figure-6-title">مختصات سلول امکان بازگشت به ناحیه واقعی سند را فراهم می‌کند؛ مقادیر صرفاً آموزشی‌اند.</title><defs><marker id="arrow-6" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="280" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="230" rx="8" stroke="#aab8b5" width="190" x="100" y="10"/><rect fill="#5f6964" height="9" rx="2" stroke="none" width="93.1" x="176.0" y="28"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="115" y="52"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="115" y="64"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="115" y="76"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="115" y="88"/><rect fill="#fffdf9" height="70" rx="2" stroke="#aab8b5" width="160" x="115" y="110"/><rect fill="#e1f1ef" height="17.5" rx="1" stroke="none" width="160" x="115" y="110"/><path d="M115 127.5H275" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M115 145.0H275" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M115 162.5H275" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M168.33333333333334 110V180" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M221.66666666666669 110V180" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="115" y="192"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="148" x="127" y="202"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="136" x="139" y="212"/><rect fill="#e1f1ef" height="17.5" rx="2" stroke="#176e73" width="53.333333333333336" x="168.33333333333334" y="127.5"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="195" y="140">۱۸٪</text><path d="M215 130L430 130" fill="none" marker-end="url(#arrow-6)" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="75" rx="8" stroke="#176e73" width="220" x="435" y="93"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="545" y="119">کادر هندسی (BBox)</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="545" y="145">[120, 340, 165, 360]</text></g></svg><figcaption>مختصات سلول امکان بازگشت به ناحیه واقعی سند را فراهم می‌کند؛ [مثال فرضی]</figcaption></figure>

در طراحی داده، واحد مختصات، مبدأ، چرخش و اندازهٔ صفحه را صریح ثبت کنید. مختصات یک Crop تصویری با مختصات PDF یکسان نیستند. اگر تصویر Resize شده یا چرخیده باشد، نگاشت آن به صفحهٔ اصلی هم باید قابل بازسازی باشد. برای سندی که بعداً کاربر روی استنادش کلیک می‌کند، این موضوع فقط یک جزئیات گرافیکی نیست.

### مطالعهٔ موردی: ۳۰۸ نویسهٔ PUA در یک صفحهٔ فارسی

**[آزمایش محدود]** در یک صفحهٔ دشوار از اسناد مالی/سازمانی فارسی، PyMuPDF ۹۸۰ نویسه استخراج کرد که ۳۰۸ نویسه از محدودهٔ Private Use Area بودند؛ حدود ۳۱٫۴٪ از نویسه‌ها. صفحه برای چشم انسان خوانا بود، ولی متن استخراج‌شده در بخش‌هایی بازنمایی معناداری از محتوای موردانتظار نبود.

PUA محدوده‌ای از Unicode برای استفادهٔ خصوصی است. حضور یک نویسهٔ PUA به‌خودی‌خود به معنی فساد نیست؛ بعضی سندها یا فونت‌ها عمداً از نگاشت خصوصی استفاده می‌کنند. مسئله زمانی است که نتوانید از آن نگاشت به متن واقعی و قابل‌استفاده برسید. در PDF، شکل ظاهری Glyph و نگاشت آن به یک نویسهٔ Unicode همیشه یک چیز نیستند. نگاشت ناقص یا نادرست می‌تواند متن قابل‌دیدن را به رشتهٔ بی‌معنی تبدیل کند. [۲](#ref-2)

**ToUnicode CMap؛ چرا شکل درست و متن غلط است؟** PDF برای رسم متن از کدهای فونت و Glyph استفاده می‌کند. نگاشت `/ToUnicode` در Font Dictionary به خواننده کمک می‌کند کدهای متن را به دنبالهٔ Unicode برگرداند. نبودن یا خرابی آن، به‌خصوص در PDFهای ساخته‌شده با فونت‌های فارسی قدیمی یا بعضی مسیرهای چاپ و TeX/XeLaTeX، می‌تواند باعث استخراج PUA، کدهای نامربوط یا متن ناقص شود. این اتفاق منحصر به فونت خاصی نیست و نبود CMap هم همیشه به معنی شکست نیست؛ Decoder ممکن است از نگاشت‌های استاندارد دیگر کمک بگیرد. در PyMuPDF فونت‌های صفحه و شیء فونت را بررسی کنید؛ دنبال `/ToUnicode` و تطابق کد با شکل واقعی بگردید. اگر امکان بازیابی نگاشت معتبر وجود دارد، اول همان را اصلاح یا Decoder دیگر را آزمایش کنید؛ اگر معنا از کدها قابل بازیابی نیست، OCR/VLM هدفمند روی تصویر شاهد مسیر جایگزین است. نرمال‌سازی نمی‌تواند نگاشت ازدست‌رفته را از هیچ بازسازی کند. [۲](#ref-2)

**روش تشخیص:** خروجی خام را نگه دارید، نرخ نویسهٔ جایگزین مانند � یا PUA را در ناحیه‌های متنی بررسی کنید، متن و هندسهٔ استخراج‌شده را کنار تصویر صفحه بگذارید و با یک Decoder دیگر مقایسه کنید. ببینید آیا مشکل محدود به یک فونت، Span یا جدول است یا کل صفحه.

**مسیر حل:** اگر PDF نگاشت معتبر دارد و فقط ترتیب متن مشکل دارد، نیازی به OCR همهٔ صفحه نیست؛ Reading Order را بررسی کنید. اگر نگاشت متن قابل‌استفاده نیست، یک OCR یا رونویسی تصویری برای ناحیهٔ خراب امتحان کنید. اگر جدول سالمی در کنار متن خراب وجود دارد، هندسهٔ سالم را دور نریزید. هدف تعمیر همان بخشی است که شواهدش ضعیف است.

**نکته:** نمرهٔ PUA یا تعداد نویسهٔ خراب یک Heuristic است؛ یعنی نشانه‌ای برای بررسی، نه اثبات خطا یا معیار پذیرش مستقل. آستانه را روی دادهٔ خودتان آزمایش کنید.

## ۵. OCR و پیش‌پردازش؛ آیا واقعاً به مدل جدید نیاز داریم؟

وقتی تصویر صفحه یا ناحیه را داریم ولی متن قابل اتکا نداریم، OCR وارد می‌شود. در این مرحله دو خطای متفاوت داریم: ممکن است Text Detector اصلاً ناحیهٔ متن را پیدا نکند، یا آن را درست پیدا کند ولی Recognizer متن را غلط بخواند. اگر کادر متن از ابتدا اشتباه است، تغییر Recognizer الزاماً کمک نمی‌کند.

### چه زمانی OCR لازم است و چه زمانی نباید آن را اجرا کنیم؟

OCR را وقتی لازم بدانید که اطلاعات متنی مورد نیاز به‌صورت پیکسل در دسترس است و نمایش دیجیتال قابل اعتماد ندارید. برای اسکن، عکس، تصویر داخل PDF و متن دیجیتالی که نگاشت آن خراب است، OCR یک مسیر منطقی است. OCR می‌تواند از موتور سنتی، مدل Recognition عمیق یا VLM تشکیل شود؛ نیازی نیست همهٔ این‌ها را پشت سر هم اجرا کنید. اگر کل صفحه یا ناحیهٔ هدف لایهٔ متن سالم دارد، اجرای دوبارهٔ OCR را پیش‌فرض نکنید. مشکل جدول را هم با OCR حل نکنید وقتی متن سلول‌ها صحیح است ولی ارتباط سطر و ستون خراب است.

دقت کنید که «OCR اجرا نشود» با «هیچ پردازش تصویر لازم نیست» فرق دارد: می‌توانید برای تشخیص Layout یا بررسی شاهد، صفحهٔ دیجیتال را رندر کنید بدون آنکه متن آن را دوباره بخوانید. برعکس، یک PDF ممکن است متن قابل انتخاب داشته باشد ولی اسکن با OCR قبلی باشد؛ در این صورت صحت لایهٔ متن را بررسی کنید. اگر فقط عکس یا لوگوی بدون متن در صفحه وجود دارد و کاربرد به آن نیازی ندارد، OCR آن ناحیه هم ممکن است اتلاف منابع باشد. اگر Caption، شمارهٔ سریال یا متن داخل همان عکس لازم است، باید آن را وارد قرارداد پوشش کنید.

برای خوانش پیکسل‌های فارسی، مخصوصاً اسناد حساس، پیشنهاد فعلی من VLM-first است؛ OCR سنتی یا مدل تخصصی سبک را برای مقایسه، کاربرد محدودِ تأییدشده یا محدودیت هزینه و زیرساخت نگه می‌دارم. برای دست‌خط فارسی، Recognition دست‌نویس (HTR) را جدا از OCR متن چاپی آزمایش کنید. یک مدل خوب روی فونت‌های فارسی ممکن است برای دست‌خط‌های آزاد یا مهر و امضا ضعیف باشد. پیش‌پردازش می‌تواند کمک کند ولی پیکسل‌هایی را که از ابتدا ثبت نشده‌اند بازیابی تضمینی نمی‌کند. اگر ورودی ناخواناست، محصول بهتر است از کاربر عکس واضح‌تر بخواهد یا به بازبینی انسانی برود. مستندات نسخهٔ ابزار مورد استفاده را نیز بخوانید؛ فعال بودن OCR در کتابخانه یا پشتیبانی زبان، به معنی فعال شدن آن برای تمام تصاویر داخل PDF نیست. [۴](#ref-4) [۳۱](#ref-31)

### تصویر خوب با تصویر مناسب OCR فرق دارد

پیش‌پردازش یعنی بررسی و در صورت نیاز اصلاح چرخش صفحه، کجی (Deskew)، کنتراست، نویز و اندازهٔ تصویر. در فارسی، نقطه‌های حروف و علائم کوچک اهمیت زیادی دارند. یک فیلتر حذف نویز ممکن است تصویر را ظاهراً تمیزتر کند، ولی نقطه، اعشار یا درصد را از بین ببرد. بنابراین نسخهٔ اصلی و نسخهٔ پردازش‌شده را جدا نگه دارید و کیفیت هر دو را بسنجید. [۴](#ref-4) [۵](#ref-5)

DPI بالاتر به معنی کیفیت بالاتر قطعی نیست. اگر اطلاعات در اسکن اولیه ثبت نشده باشند، بزرگ کردن تصویر آن‌ها را برنمی‌گرداند. از یک تنظیم معمول برای رندر شروع کنید و چند رزولوشن را روی نمونه‌های یکسان مقایسه کنید. برای متن ریز، Crop با حاشیهٔ مناسب و رندر دقیق‌تر را امتحان کنید؛ برای صفحهٔ کامل، هزینهٔ حافظه و زمان اجرا را هم در نظر بگیرید.

| ابزار | مسئولیت اصلی | وضعیت شواهد در این مقاله |
|---|---|---|
| Tesseract | موتور OCR با فایل‌های زبان؛ فارسی با کد `fas` | گزینهٔ مستند برای Baseline؛ بدون ادعای نتیجهٔ آزمایش شخصی |
| PaddleOCR / PP-OCR | تشخیص و خواندن متن؛ مدل‌های چندزبانه از جمله گزینهٔ فارسی در نسخه‌های مستند | در فهرست گزینه‌های بررسی؛ بدون نتیجهٔ قطعی |
| OCRmyPDF | افزودن لایهٔ جست‌وجو به PDF با OCR و پیش‌پردازش | گزینهٔ پژوهش عمومی؛ به‌خودی‌خود بازسازی کامل جدول نیست |
| Bina Rizeh | مدل OCR فارسی که روی Cropهای واقعی آزمایش شد | نتیجهٔ محدود و ناموفق برای معیارهای حساس همان آزمایش؛ بخش ۸ |
| VLMها | رونویسی تصویری یا بازسازی پیچیده‌تر بر اساس تصویر و دستور | مسیر اولیهٔ پیشنهادی برای تصاویر فارسی حساس؛ بخش ۱۰ |

پشتیبانی از زبان فارسی در مستندات یک موتور به معنی قبولی روی اسناد فارسی شما نیست. ممکن است OCR روی صفحهٔ کتاب فارسی عالی باشد ولی روی فاکتور، جدول، فونت اداری، عدد لاتین یا نماد ریاضی مشکل داشته باشد. مدل و فایل زبان دقیق، نسخهٔ موتور و تنظیمات را ثبت کنید. [۴](#ref-4) [۵](#ref-5) [۶](#ref-6)

**چه زمانی OCR جدید امتحان کنیم؟** وقتی با تصویر سالم و Crop درست، خروجی خام Recognizer هنوز خطای تکرارشونده دارد. اگر متن درست خوانده شده و بعداً در جدول یا RTL خراب می‌شود، به لایهٔ بعدی بروید. این تفکیک جلوی خیلی از آزمایش‌های بی‌هدف را می‌گیرد.

### یک سؤال واقعی از کامیونیتی: فاکتورهای دست‌نویس و عکس‌های بی‌کیفیت را چطور بخوانیم؟

در بحث فاکتورهای دست‌نویس (۱۱ اکتبر ۲۰۲۶)، پیشنهاد اولیهٔ من Gemini و بهبود عکس بود؛ اشارهٔ یکی از اعضا به قالب ثابت، سؤال را دقیق‌تر کرد: آیا محتوا دیجیتال است یا تصویری؟ آیا فاکتورها قالب ثابت دارند؟

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-7-title" direction="ltr" height="378" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 378" width="720" xml:lang="fa"><title id="v5-figure-7-title">انتخاب مسیر پردازش فاکتور بر اساس نوع ورودی</title><rect x="0" y="0" width="720" height="378" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="22" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="25">استخراج فاکتور بر اساس نوع ورودی و تنوع</text><rect fill="#e1f1ef" height="50" rx="10" stroke="#176e73" width="340" x="190" y="44"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="360" y="75">متن دیجیتال معتبر است؟</text><path d="M360 94V116H180V138M360 116H540V138" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="180" y="111">بله</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="540" y="111">خیر / اسکن</text><rect fill="#fffdf9" height="75" rx="10" stroke="#aab8b5" width="330" x="15" y="138"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="180" y="166">استخراج Native</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="180" y="191">قالب ثابت / نگاشت برچسب</text><rect fill="#fffdf9" height="75" rx="10" stroke="#aab8b5" width="330" x="375" y="138"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="540" y="166">کیفیت تصویر و چیدمان</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="540" y="191">فارسی: VLM-first؛ مقایسهٔ OCR</text><path d="M180 213V245H360M540 213V245H360V265m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#e1f1ef" height="68" rx="10" stroke="#176e73" width="430" x="145" y="265"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="292">اعتبارسنجی مبلغ، واحد و ردیف</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="315">بازبینی فیلد نامعلوم با تصویر</text></g></svg><figcaption>معماری پیشنهادی برای فاکتور: ساده‌ترین مسیر مناسب را انتخاب کنید و در پایان صحت فیلدها و روابط را بررسی کنید. این نمودار گزارش اجرای Benchmark نیست. مسیر پیشنهادی برای فارسی حساس: VLM-first برای خوانش بصری؛ OCR تخصصی برای مقایسه یا کاربرد تأییدشده.</figcaption></figure>

| وضعیت ورودی | مسیر پیشنهادی برای آزمایش | چه چیزی را بسنجیم؟ |
|---|---|---|
| فاکتور دیجیتال با قالب ثابت | Native Extraction و نگاشت فیلدها براساس مختصات یا برچسب؛ نیازی به VLM نیست اگر نتیجه معتبر باشد | مبلغ، واحد، شمارهٔ فاکتور و تغییر قالب |
| تصویر فاکتور با قالب ثابت | تصحیح پرسپکتیو و چرخش، تشخیص ناحیه یا Template Alignment، سپس خوانش هدفمند هر فیلد؛ برای فارسی ابتدا VLM و مقایسه با OCR تخصصی | دقت کادر، خوانش اعداد و مقاومت در برابر تغییر مقیاس |
| عکس با قالب‌های متفاوت | Layout Detection، خوانش ناحیه‌ها و بازسازی ارتباط عنوان و مقدار؛ VLM به‌عنوان موتور اصلی پیشنهادی برای فارسی | فیلدهای جاافتاده، اتصال مبلغ به قلم درست و هزینه |
| دست‌خط، مهر روی متن یا عکس تار | بررسی وضوح واقعی تصویر، VLM دارای قابلیت خوانش دست‌خط، مقایسه با HTR تخصصی، بازبینی انسانی برای فیلدهای حساس | نرخ خطای دست‌خط، موارد ناخوانا، پذیرش اشتباه |

مثلاً اگر مجموع فاکتور با جمع ردیف‌ها جور درنمی‌آید، به‌طور خودکار یکی از اعداد را تغییر ندهید تا معادله درست شود. ممکن است تخفیف یا مالیات وجود داشته باشد، یا یکی از مقادیر اشتباه خوانده شده باشد. جمع را به‌عنوان سیگنال تشخیص تعارض استفاده کنید و شاهد تصویری فیلدها را بررسی کنید. اگر کیفیت عکس اطلاعات لازم را از بین برده، افزایش وضوح مصنوعی یا Super-resolution ممکن است جزئیات ظاهری تازه تولید کند، نه شواهد معتبر. گرفتن عکس دوباره یا اعلام ناخوانا بودن گاهی از هر مدل جدیدی بهتر است.

**[مسیر پیشنهادی]**

## ۶. Layout Analysis و Reading Order؛ متن را کجا قرار بدهیم؟

فرض کنید یک گزارش دو ستون دارد. عنوان بالای صفحه، متن ستون راست، متن ستون چپ، شکل و Caption در پایین صفحه دیده می‌شوند. اگر فقط مختصات عمودی کلمات را مرتب کنید، ممکن است سطرهای دو ستون با هم ترکیب شوند. برای یک سند فارسی، حتی این فرض که ستون چپ قبل از ستون راست خوانده می‌شود هم ممکن است اشتباه باشد.

Layout Model به ما کمک می‌کند ناحیه‌های معنایی را پیدا کنیم. Reading Order مسئلهٔ دیگری است: تعیین توالی خواندن آن‌ها. ممکن است مدل همهٔ ناحیه‌ها را درست تشخیص دهد ولی ترتیب اشتباه تولید کند. [۱](#ref-1) [۳](#ref-3) [۷](#ref-7)

### Docling و TableFormer چه نقشی دارند؟

Docling یک چارچوب تبدیل سند به نمایش ساختاریافته است. در مسیرهای PDF می‌تواند از مدل Layout، موتورهای OCR، تشخیص ساختار جدول و سایر اجزای قابل تنظیم استفاده کند. `DoclingDocument` نمونه‌ای از نمایش ساختاریافته‌ای است که اجزای سند، روابط و Provenance را نگه می‌دارد. TableFormer در این معماری برای تشخیص ساختار جدول است؛ Parser مستقلی نیست که لازم باشد بعد از Docling دوباره تمام سند را از صفر بخواند. [۳](#ref-3)

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-8-title" direction="ltr" height="258" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 258" width="720" xml:lang="fa"><title id="v5-figure-8-title">تصویر شماتیک از وظیفه TableFormer؛ خروجی واقعی مدل نیست.</title><defs><marker id="arrow-8" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="258" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="140" rx="2" stroke="#aab8b5" width="245" x="65" y="40"/><rect fill="#e1f1ef" height="28.0" rx="1" stroke="none" width="245" x="65" y="40"/><path d="M65 68.0H310" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M146.66666666666669 96.0H310" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M146.66666666666669 124.0H310" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M146.66666666666669 152.0H310" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M146.66666666666669 40V180" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M228.33333333333334 40V180" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#fffdf9" height="140" rx="2" stroke="#aab8b5" width="245" x="410" y="40"/><rect fill="#e1f1ef" height="28.0" rx="1" stroke="none" width="245" x="410" y="40"/><path d="M410 68.0H655" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M491.6666666666667 96.0H655" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M491.6666666666667 124.0H655" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M491.6666666666667 152.0H655" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M491.6666666666667 40V180" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M573.3333333333334 40V180" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M325 110L395 110" fill="none" marker-end="url(#arrow-8)" stroke="#176e73" stroke-width="2"/><rect fill="none" height="104" rx="2" stroke="#176e73" width="70" x="415.0" y="71"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="496.6" y="71"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="496.6" y="99"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="496.6" y="127"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="496.6" y="155"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="578.2" y="71"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="578.2" y="99"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="578.2" y="127"/><rect fill="none" height="20" rx="2" stroke="#176e73" width="70" x="578.2" y="155"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="187" y="210">جدول اصلی</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="532" y="210">هندسه و سلول‌های تشخیص‌داده‌شده</text></g></svg><figcaption>[مثال شماتیک] TableFormer: یک کادر پیوسته برای سلول چهارسطری، هشت کادر برای سلول‌های مستقل بدنه؛ نه خروجی واقعی مدل.</figcaption></figure>

در یکی از مسیرهایی که بررسی کردم، تنظیمات TableFormer در Docling با OCR غیرفعال آزمایش شد. این انتخاب برای آزمایش متن Native و ساختار جدول بود، نه توصیهٔ عمومی برای تمام PDFها.

| گزینهٔ ثبت‌شده | چه چیزی را تغییر می‌دهد؟ |
|---|---|
| `do_table_structure` | فعال کردن تشخیص ساختار جدول |
| `ACCURATE` | حالت TableFormer با اولویت دقت در مقابل سرعت |
| `do_cell_matching` | استفاده یا عدم استفاده از تطبیق متن سلول‌ها با متن PDF، مطابق رفتار نسخهٔ ابزار |
| `do_ocr` | OCR در آن مسیر آزمایشی غیرفعال بود |

در صفحهٔ دشوار، تغییر Cell Matching به‌تنهایی مشکل Unicode خراب را حل نکرد. ساختار جدول هم ناقص ماند و یک فرمول تصویری تشخیص داده نشد. چرا؟ چون یک تنظیم مربوط به اتصال سلول‌ها لزوماً لایهٔ Recognition یا Coverage را اصلاح نمی‌کند. نام حالت `ACCURATE` هم تضمین نمی‌کند که خروجی روی دادهٔ شما درست باشد.

برای پیدا کردن ریشهٔ مشکل، سه خروجی را جدا نگاه کنید: کادرهای Layout روی تصویر، متن خام استخراج‌شده و ترتیب نهایی عناصر. اگر فرمول یا Caption اصلاً کادر ندارد، Detection یا طبقه‌بندی را بررسی کنید؛ اگر کادر درست است و متن ناقص است، مسیر Recognition را بررسی کنید؛ اگر هر دو درست‌اند ولی محتوا در جای اشتباه ظاهر می‌شود، سراغ Reading Order یا Reconciliation بروید.

### Docling، SmolDocling، Granite-Docling و DocTags؛ چهار اسم با مسئولیت متفاوت

در پستی که ۲۷ سپتامبر ۲۰۲۵ در کانال منتشر کردم، دربارهٔ Granite-Docling و مدل‌های کوچک تخصصی سند نوشتم. نکتهٔ آموزشی مهم این است که انتخاب Framework با انتخاب مدل داخل Framework یکی نیست.

| نام | ماهیت | وظیفه |
|---|---|---|
| Docling | کتابخانه و چارچوب پردازش سند | مدیریت ورودی، اجرای Pipeline، نمایش ساختاریافته و تولید خروجی |
| TableFormer | مدل تخصصی ساختار جدول | تشخیص هندسه و روابط سلول‌ها در مسیرهای مربوط |
| SmolDocling-256M-preview | مدل کوچک Vision-Language برای تبدیل صفحه | تولید نمایش نشانه‌گذاری‌شدهٔ ساختار و محتوا |
| Granite-Docling-258M | مدل تخصصی IBM برای تبدیل سند | مسیر مدل‌محور جایگزین چند وظیفهٔ جدا، با خروجی DocTags |
| DocTags | نمایش نشانه‌گذاری‌شده | توصیف نوع عناصر، متن، مکان و بخشی از روابط ساختاری؛ ورودی قابل تبدیل به DoclingDocument |

Granite-Docling-258M با معماری مبتنی بر Idefics3، رمزگذار تصویری SigLIP2 و مدل زبانی کوچک Granite معرفی شده و مجوز مدل Apache 2.0 است. خروجی DocTags می‌تواند با ابزارهای Docling به Markdown، HTML یا نمایش ساختاریافته تبدیل شود. مدل در کارت رسمی‌اش برای انگلیسی معرفی شده و پشتیبانی ژاپنی، عربی و چینی آزمایشی است؛ حتی پشتیبانی عربی هم به معنی تأیید فارسی نیست. [۱۶](#ref-16)

چرا این رویکرد ارزش بررسی دارد؟ چون به‌جای درخواست آزاد از یک مدل بزرگ، می‌توان یک مدل کوچک را برای تبدیل صفحه به نمایشی محدود و مناسب سند ارزیابی کرد. اما خروجی ساختاریافتهٔ تولیدشده همچنان ممکن است ناحیه‌ای را حذف کند یا ترتیب و مقدار را اشتباه بگیرد. DocTags یک قرارداد نمایش است، نه ضمانت صحت معنایی و نه استاندارد اجباری همهٔ Parserها. اگر آن را به DoclingDocument تبدیل می‌کنید، بررسی کنید رابطه‌های مورد نیازتان، مختصات و شواهد واقعاً تا خروجی نهایی حفظ شده‌اند.

برای مقایسهٔ منصفانه، مسیر کلاسیک Docling با مدل‌های جداگانه را کنار مسیر VLM تخصصی (مثل Granite-Docling) روی همان صفحه‌ها بگذارید. دقت جدول، پوشش، ترتیب خواندن، متن فارسی و مصرف حافظه را جدا بسنجید. [دموی Granite-Docling](https://huggingface.co/spaces/ibm-granite/granite-docling-258m-demo) برای آشنایی مناسب است؛ نه برای ارسال اسناد محرمانه یا اثبات آمادگی Production.

### چه ابزارهای دیگری برای مقایسه ارزش دارند؟

Surya برای کارهایی مانند OCR، Layout و Reading Order گزینهٔ قابل بررسی است. PP-DocLayout و PP-Structure در اکوسیستم PaddleOCR نیز به تشخیص ناحیه‌ها و بازسازی سند مربوط‌اند. Marker و MinerU نمونه‌هایی از Pipelineهای سطح سند هستند، نه فقط OCR. کارایی دقیق هر کدام تابع نسخه، تنظیمات، Backend و داده است. [۶](#ref-6) [۷](#ref-7) [۸](#ref-8)

برای انتخاب، ابتدا همان صفحه‌ها و برچسب‌های مرجع را به ابزارهای مختلف بدهید. بررسی کنید آیا عنوان، جدول، شکل، فرمول، پاورقی، فهرست و فرم دیده شده‌اند. بعد سراغ متن و ترتیب بروید. اگر صرفاً Markdown نهایی را با چشم مقایسه کنید، حذف ناحیه‌های کوچک ممکن است اصلاً به چشمتان نیاید.

### یک گزینهٔ Managed: Gemini Layout Parser در Google Document AI

در کانال، در ۲۲ اکتبر ۲۰۲۶ به Gemini Layout Parser اشاره کردم. آن را با فراخوانی مستقیم Gemini API برای خواندن PDF اشتباه نگیرید. این سرویس در Google Document AI برای ترکیب OCR، تشخیص عناصر ساختاری و غنی‌سازی Chunkها طراحی شده است. می‌تواند ساختار عنوان‌ها و جدول‌ها را در خروجی Chunk لحاظ کند و برای بعضی عناصر بصری توضیح تولید کند. این توضیح تولیدشده باید از متن مستند سند قابل تفکیک باشد؛ توضیح یک نمودار را به‌جای اعداد اصلی نمودار وارد دادهٔ معتبر نکنید. [۱۷](#ref-17)

در مستندات بررسی‌شده در ۲۶ نوامبر ۲۰۲۶، سقف پردازش آنلاین ۲۰ مگابایت و ۱۵ صفحه برای PDF، و سقف Batch برای PDF یک گیگابایت و ۵۰۰ صفحه ذکر شده است. پشتیبانی عمومی PDF، HTML، DOCX، PPTX، XLSX و XLSM مشمول هزینه است. مهم‌تر از این اعداد، محدودیت ساختاری است: جدول چندصفحه‌ای ممکن است به دو جدول جدا تبدیل شود. این محدودیت‌ها مربوط به سرویس و حالت پردازش مستندشده‌اند، نه محدودیت عمومی Gemini API. در فهرست Processorها، برای حالت آنلاین `imageless_mode` سقف ۳۰ صفحه ذکر شده است؛ این حالت را با مسیر تصویری یکسان فرض نکنید. طبق فهرست بررسی‌شده، `pretrained-layout-parser-v3.5-2026-07-09` نسخهٔ GA مبتنی بر Gemini 3.5 Flash است و `pretrained-layout-parser-v3.1-lite-2026-08-11` در Preview قرار دارد؛ نسخهٔ پیش‌فرض، نسخه‌های قابل استفاده و محل پردازش می‌توانند متفاوت باشند. برای نمونه، مستندات دو نسخهٔ 1.6 مبتنی بر Gemini 3.0 را به مسیر Global و محدودیت Data Residency مرتبط می‌کنند. حتی وقتی `fa` در فهرست زبان‌هاست، صحت فارسی یا جدول شما هنوز باید اندازه‌گیری شود. پیش از خرید یا استقرار، شناسهٔ Processor، منطقه، قیمت و مستندات جاری را بررسی کنید. [۴۳](#ref-43)

اگر سرویس آماده انتخاب می‌کنید، فقط به زمان رسیدن به اولین خروجی نگاه نکنید. ببینید آیا می‌توانید به سلول و صفحهٔ اصلی برگردید، ناحیه‌ای را دوباره پردازش کنید، جدول چندصفحه‌ای را بررسی کنید و خروجی نسخه‌دار داشته باشید. ادعای بهبود Retrieval از طرف ارائه‌دهنده، جای ارزیابی روی اسناد فارسی خودتان را نمی‌گیرد.

## ۷. Table Parsing؛ چرا خواندن عددها کافی نیست؟

اگر یک جدول ساده فقط سطر و ستون منظم دارد، روش‌های هندسی مبتنی بر خطوط یا موقعیت متن می‌توانند Baseline خوبی باشند. ابزارهایی مثل PyMuPDF، pdfplumber و Camelot را برای PDFهای دیجیتال بررسی کنید؛ جدول اسکن‌شده، بدون خط یا دارای سلول ادغام‌شده به روش‌های دیگری نیاز دارد. TableFormer و مدل‌های تشخیص ساختار جدول از گزینه‌های مرحلهٔ بعد هستند. [۲](#ref-2) [۳](#ref-3) [۱۳](#ref-13)

برای اینکه مسئله روشن شود، چند خانوادهٔ خطا را جدا مرور کنیم.

### سلول‌های ادغام‌شده: یک مقدار برای چند سطر

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-9-title" direction="ltr" height="203" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 203" width="720" xml:lang="fa"><title id="v5-figure-9-title">یک سلول به سه سطر تعلق دارد (rowspan).</title><rect x="0" y="0" width="720" height="203" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="150" rx="2" stroke="#aab8b5" width="550" x="85" y="10"/><rect fill="#e1f1ef" height="37.5" rx="1" stroke="none" width="550" x="85" y="10"/><path d="M85 47.5H635" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M85 85.0H451.6666666666667" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M85 122.5H451.6666666666667" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M268.33333333333337 10V160" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M451.6666666666667 10V160" fill="none" stroke="#aab8b5" stroke-width="1"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="543" y="38">نوع قرارداد</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="38">مدت</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="177" y="38">نرخ سود</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="543" y="110">فروش اقساطی</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="76.0">۱۲ ماه</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="177" y="76.0">۱۸٪</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="113.5">۲۴ ماه</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="177" y="113.5">۲۰٪</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="151.0">۳۶ ماه</text><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="177" y="151.0">۲۳٪</text></g></svg><figcaption>[مثال فرضی] جدول راست‌به‌چپ: ستون نوع قرارداد در سمت راست است و سلول فروش اقساطی سه سطر را پوشش می‌دهد (rowspan=3).</figcaption></figure>

اگر «فروش اقساطی» در یک سلول عمودی ادغام شده و به سه ردیف نرخ تعلق دارد، باید این ارتباط حفظ شود. کپی کردن یک متن در سه سطر شاید برای خروجی جست‌وجو مفید باشد، اما ساختار اصلی باید `rowspan` را هم نگه دارد. در غیر این صورت موقع بازسازی جدول یا Chunking نمی‌دانیم آن سه مقدار به یک قرارداد مربوط‌اند.

**روش بررسی:** کادر سلول و سطرها را روی تصویر بیندازید، پوشش شبکه را کنترل کنید و ببینید هر موقعیت جدول به کدام سلول منطقی تعلق دارد. خانهٔ خالی، خانهٔ پوشیده‌شده توسط Span و خانهٔ ناخوانا یک چیز نیستند.

### جدول چندصفحه‌ای: سرستون را از کجا بیاوریم؟

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-10-title" direction="ltr" height="233" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 233" width="720" xml:lang="fa"><title id="v5-figure-10-title">جدول ادامه‌دار و سرستون ارث‌رسیده از صفحه قبل.</title><defs><marker id="arrow-10" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="233" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="145" rx="2" stroke="#aab8b5" width="245" x="55" y="40"/><rect fill="#e1f1ef" height="36.25" rx="1" stroke="none" width="245" x="55" y="40"/><path d="M55 76.25H300" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M55 112.5H300" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M55 148.75H300" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M136.66666666666669 40V185" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M218.33333333333334 40V185" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#fffdf9" height="145" rx="2" stroke="#aab8b5" width="245" x="420" y="40"/><path d="M420 76.25H665" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M420 112.5H665" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M420 148.75H665" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M501.6666666666667 40V185" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M583.3333333333334 40V185" fill="none" stroke="#aab8b5" stroke-width="1"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="177" y="25">صفحه ۱۵</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="542" y="25">صفحه ۱۶</text><path d="M312 108L408 108" fill="none" marker-end="url(#arrow-10)" stroke="#176e73" stroke-width="2"/></g></svg><figcaption>جدول ادامه‌دار و سرستون ارث‌رسیده از صفحه قبل.</figcaption></figure>

گاهی صفحهٔ بعد ادامهٔ یک جدول است ولی سرستون ندارد. گاهی هم جدول دیگری با تعداد ستون‌های مشابه شروع می‌شود. برای ادغام باید سرستون‌ها، هم‌ترازی ستون‌ها، نوع داده، عنوان و بخش سند، نشانه‌های ادامهٔ جدول و مرز صفحات را با هم بررسی کنید. سطر شکسته‌شده بین دو صفحه نیز با سرستون تکراری تفاوت دارد.

در یکی از نمونه‌های واقعی، اتصال احتمالی دو صفحه شناسایی شد اما سلول‌های تأییدنشده و سرستون‌های ناقص اجازهٔ پذیرش جدول نهایی را ندادند. **شناسایی یک رابطهٔ احتمالی با اثبات آن فرق دارد.**

### دو جدول مجاور با معنای متفاوت

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-11-title" direction="ltr" height="213" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 213" width="720" xml:lang="fa"><title id="v5-figure-11-title">دو جدول مجاور باید از نظر ساختاری و معنایی جدا بمانند.</title><rect x="0" y="0" width="720" height="213" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="120" rx="2" stroke="#aab8b5" width="300" x="30" y="40"/><rect fill="#e1f1ef" height="30.0" rx="1" stroke="none" width="300" x="30" y="40"/><path d="M30 70.0H330" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M30 100.0H330" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M30 130.0H330" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M180.0 40V160" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#fffdf9" height="120" rx="2" stroke="#aab8b5" width="300" x="390" y="40"/><rect fill="#e1f1ef" height="30.0" rx="1" stroke="none" width="300" x="390" y="40"/><path d="M390 70.0H690" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M390 100.0H690" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M390 130.0H690" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M540.0 40V160" fill="none" stroke="#aab8b5" stroke-width="1"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="180" y="25">شرایط ضمانت</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="540" y="25">شرایط تسهیلات</text><path d="M360 10V170" fill="none" stroke="#94600d" stroke-width="2"/></g></svg><figcaption>دو جدول مجاور باید از نظر ساختاری و معنایی جدا بمانند.</figcaption></figure>

تصور کنید در یک صفحه دو جدول مستقل دربارهٔ شرایط ضمانت و شرایط دریافت خدمات وجود دارند. ممکن است فاصلهٔ هندسی‌شان کم و ظاهرشان مشابه باشد. ادغام آن‌ها یک جدول بزرگ و ظاهراً مرتب تولید می‌کند، اما داده را از نظر معنایی خراب می‌کند.

برای جدا نگه داشتن آن‌ها، ابتدا ناحیهٔ جدول، Caption، عنوان بخش و پاورقی را ثبت کنید. ادغام براساس فاصله یا تعداد ستون به‌تنهایی کافی نیست. در بازبینی، علاوه بر ساختار داخل جدول، مرز معنایی بین جدول‌ها را هم بسنجید.

### فرم خالی، جدول خراب یا دادهٔ واقعاً ناموجود؟

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-12-title" direction="ltr" height="308" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 308" width="720" xml:lang="fa"><title id="v5-figure-12-title">ساختار یک فرم خالی، حتی بدون متن، باید حفظ شود.</title><defs><marker id="arrow-12" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="308" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="230" rx="2" stroke="#aab8b5" width="235" x="70" y="10"/><path d="M70 22.77777777777778H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 35.55555555555556H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 48.333333333333336H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 61.111111111111114H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 73.88888888888889H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 86.66666666666667H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 99.44444444444444H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 112.22222222222223H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 125.0H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 137.77777777777777H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 150.55555555555554H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 163.33333333333334H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 176.11111111111111H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 188.88888888888889H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 201.66666666666666H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 214.44444444444446H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M70 227.22222222222223H305" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M117.0 10V240" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M164.0 10V240" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M211.0 10V240" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M258.0 10V240" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#fffdf9" height="35" rx="2" stroke="#aab8b5" width="210" x="450" y="95"/><path d="M492.0 95V130" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M534.0 95V130" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M576.0 95V130" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M618.0 95V130" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M320 115L435 115" fill="none" marker-end="url(#arrow-12)" stroke="#176e73" stroke-width="2"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="187" y="262">۱۸ × ۵ = ۹۰ جایگاه</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="555" y="154">بازسازی ناقص</text></g></svg><figcaption>ساختار یک فرم خالی، حتی بدون متن، باید حفظ شود.</figcaption></figure>

یک فرم خالی ممکن است اصلاً مقدار نداشته باشد ولی ساختار فیلدها و محل پاسخ مهم باشد. اگر خروجی خالی را همیشه شکست OCR فرض کنید، اطلاعاتی را که واقعاً وجود ندارد ممکن است با حدس تولید کنید. برعکس، یک سلول با نوشتهٔ ریز ممکن است به‌اشتباه خالی تشخیص داده شود. برای خانهٔ خالی، ادغام‌شده، ناخوانا و حذف‌شده وضعیت جدا داشته باشید.

### سرستون، واحد و پاورقی؛ رابطه‌های کوچک با اثر بزرگ

فرض کنید بالای یک جدول نوشته شده «مبالغ به میلیون ریال» یا کنار نرخ‌ها علامت ستاره‌ای وجود دارد که شرطی را در پاورقی مشخص می‌کند. اگر این ارتباط از بین برود، همهٔ سلول‌ها می‌توانند کاملاً درست OCR شده باشند و باز هم خروجی معنای غلطی داشته باشد. سرستون چندسطحی، عنوان جدول و واحد باید جزء ساختار منطقی باشند، نه صرفاً رشته‌هایی نزدیک جدول.

برای ارزیابی، حداقل دو محور جدا داشته باشید: **درستی ساختار جدول** و **درستی محتوای سلول‌ها**. سپس رابطهٔ مقدار با سرستون و واحد را بررسی کنید. یک معیار میانگین برای تمام متن صفحه، این سه نوع خطا را به‌خوبی از هم جدا نمی‌کند.

## ۸. فارسی؛ چالش فقط راست‌به‌چپ بودن نیست

در طراحی یک Parser فارسی باید سه نوع خطا را از هم جدا کنیم: **خطای بازنمایی متن در فایل، خطای خواندن نویسه‌ها از تصویر و خطای بازسازی ترتیب و معنی.** ممکن است هر سه هم‌زمان در یک سند وجود داشته باشند. اگر همه را «مشکل RTL» بنامیم، احتمالاً راهکار اشتباهی انتخاب می‌کنیم.

### Unicode، فونت و نیم‌فاصله

در فارسی شکل یک حرف بسته به موقعیتش در کلمه تغییر می‌کند؛ اما در متن منطقی قرار نیست برای هر شکل ظاهری، یک حرف مستقل ذخیره کنیم. در استخراج PDF ممکن است به‌جای متن مورد انتظار با Glyph ID، نویسهٔ PUA یا توالی‌های غیرمعمول روبه‌رو شویم. اول باید بدانیم خروجی واقعاً چه Code Pointهایی دارد؛ صرفاً به چیزی که مرورگر نشان می‌دهد نگاه نکنیم.

از طرف دیگر، متنی که درست خوانده شده هم ممکن است برای جست‌وجو به نرمال‌سازی نیاز داشته باشد: مثلاً «ي» و «ی» یا «ك» و «ک» دو نویسهٔ متفاوت‌اند. نیم‌فاصله (ZWNJ) هم در فارسی کاربرد واقعی دارد. یکی کردن بعضی نویسه‌ها برای جست‌وجو می‌تواند مفید باشد، اما حذف کورکورانهٔ همهٔ فاصله‌ها و علائم راه درستی برای اصلاح متن نیست.

پیشنهاد من نگهداری حداقل دو نمایش است: متن خامِ متصل به منبع و متن نرمال‌شده برای کاربرد مشخص. اگر نرمال‌سازی روی طول یا ترتیب متن اثر می‌گذارد، رابطهٔ آن با متن خام را هم حفظ کنید. نرمال‌سازی Unicode، نگاشت‌های زبانی، اصلاح OCR و تبدیل اعداد چهار کار متفاوت‌اند؛ نباید در یک تابع مبهم ادغام شوند. [۹](#ref-9)

**Presentation Forms با حروف منطقی فرق دارند.** PDF Decoder گاهی شکل‌های موضعی حرف (ابتدایی، میانی، پایانی یا جدا) را از بلوک‌های Arabic Presentation Forms-A (`U+FB50–U+FDFF`) و B (`U+FE70–U+FEFF`) بیرون می‌دهد. همهٔ نویسه‌های داخل این بازه‌ها شکل موضعی نیستند (مثلاً `U+FEFF` کاربرد دیگری دارد). NFC فقط هم‌ارزی canonical را یکسان می‌کند؛ شکل سازگاری مانند `U+FEB3` (شکل ابتدایی سین، ﺳ) با NFC به `U+0633` (س) تبدیل نمی‌شود، ولی NFKC آن را برمی‌گرداند. بهتر است نگاشت هدفمند و آزموده‌شدهٔ Presentation Forms را روی نمای جست‌وجو اعمال کنید. اجرای کورکورانهٔ NFKC بر کل متن می‌تواند توان‌ها، زیرنویس‌ها، بعضی نمادهای ریاضی، لیگاتور `﷼` (که به «ریال» باز می‌شود) و تمایزهای مهم دیگر را تغییر دهد. برای رشتهٔ خام و فیلد حساس قواعد جدا داشته باشید. [۹](#ref-9)

**کشیده (Kashida/Tatweel):** در بعضی نامه‌ها و روزنامه‌های فارسی برای تراز کردن سطر از `U+0640` (ـ) استفاده می‌شود. «ســــلام» و «سلام» ممکن است از نظر جست‌وجوی واژه‌ای (BM25) دو رشتهٔ متفاوت باشند. در نمای جست‌وجوی نثر، حذف هدفمند Tatweel را با تست تطابق واژه آزمایش کنید؛ در متن خام، نقل‌قول دقیق و فرمول‌ها اصل نویسه را حفظ کنید. نیم‌فاصله نیز سیاست جداگانه می‌خواهد: حذف عمومی آن ممکن است مرز واژه را عوض کند. قبل و بعد هر تبدیل، نمونه‌های فارسی با رقم، واحد پول و فرمول را بررسی کنید.

### Visual Order در مقابل Logical Order

فارسی راست‌به‌چپ است، اما رقم‌ها، عبارت‌های لاتین و بخش‌هایی از فرمول‌ها از قواعد خودشان پیروی می‌کنند. عبارت «مبلغ ۱۲٫۵۰ USD» را در نظر بگیرید. اگر کل رشته را معکوس کنیم، هم کلمه‌ها و هم ترتیب رقم‌ها خراب می‌شوند.

دو جهت تبدیل را از هم جدا کنید:

- **Visual-to-Logical Reconstruction:** بعضی PDFها Glyphها را به ترتیب جایگاه فیزیکی رسم یا ذخیره می‌کنند. فرض کنید Decoder دنباله‌ای با کدهای متناظر `1234 مالس` (نمایش ایزولهٔ ترتیب خام، نه جملهٔ درست) تحویل می‌دهد؛ هدف بازسازی ترتیب منطقی واژه‌ها و رقم‌ها بر اساس مختصات، جهت Span، نگاشت فونت و شاهد تصویر است. این کار یک وارون‌سازی عمومی رشته نیست؛ بازسازی بصری به منطقی ممکن است مبهم باشد.
- **Logical-to-Visual Rendering:** وقتی متن منطقی صحیح است، Unicode Bidirectional Algorithm (UAX #9) ترتیب نمایش را برای UI تعیین می‌کند. `python-bidi` یا CSS راست‌به‌چپ را روی متن منطقی به‌عنوان «اصلاح» دوباره اعمال نکنید؛ ذخیره کردن خروجی بصری به‌جای متن منطقی می‌تواند پردازش بعدی را خراب کند.

Reading Order صفحه مسئلهٔ سوم است: ترتیب ستون‌ها، بلوک‌ها و سلول‌ها. ابتدا Code Point خام، سپس ترتیب عناصر، و در نهایت نمایش UI را تست کنید. [۹](#ref-9) [۳۲](#ref-32)

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-13-title" direction="ltr" height="381" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 381" width="720" xml:lang="fa"><title id="v5-figure-13-title">سه سطح مستقل در پردازش متن راست‌به‌چپ</title><rect x="0" y="0" width="720" height="381" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="94" rx="10" stroke="#aab8b5" width="680" x="20" y="15"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="41">متن منطقی Unicode</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="65">نویسه‌ها، رقم‌ها و علامت‌ها</text><text direction="rtl" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="360" y="90">بررسی کدهای خام نویسه</text><path d="M360 109V127m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="94" rx="10" stroke="#aab8b5" width="680" x="20" y="130"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="156">ترتیب نمایش (BiDi)</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="180">واژهٔ راست‌به‌چپ و رقم چپ‌به‌راست</text><text direction="rtl" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="360" y="205">اصلاح نمایش و جداسازی جهت</text><path d="M360 224V242m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="94" rx="10" stroke="#aab8b5" width="680" x="20" y="245"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="271">ترتیب خواندن سند</text><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="295">ستون‌ها، بلوک‌ها و جدول‌ها</text><text direction="rtl" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="360" y="320">اصلاح چیدمان و روابط ساختاری</text></g></svg><figcaption>سه سطح مستقل خطا: رشتهٔ Unicode، ترتیب نمایش BiDi و ترتیب خواندن اجزای سند. قبل از اصلاح، لایهٔ مشکل‌دار را پیدا کنید.</figcaption></figure>

| مقدار صحیح (مثال فرضی) | خطای احتمالی | پیامد |
|---|---|---|
| <span dir="ltr">`۱۴۰۱`</span> | <span dir="ltr">`۱۰۴۱`</span> | سال اشتباه |
| <span dir="ltr">`۹٬۰۳۹٬۰۰۰`</span> | <span dir="ltr">`۰۰۰٬۹۳۰٬۹`</span> | ترتیب رقم یا گروه‌بندی غلط |
| <span dir="ltr" style="unicode-bidi:isolate;white-space:nowrap">`+10`</span> | <span dir="ltr" style="unicode-bidi:isolate;white-space:nowrap">`10+`</span> | جابه‌جایی علامت؛ نیازمند بررسی بافت |
| <span dir="ltr" style="unicode-bidi:isolate;white-space:nowrap">`≤ 20`</span> | <span dir="ltr" style="unicode-bidi:isolate;white-space:nowrap">`< 20`</span> | تغییر شرط، با وجود شباهت ظاهری |
| <span dir="ltr">`۱۸٪`</span> | <span dir="ltr">`۱۸`</span> | حذف واحد درصد |
| ۵۰۰ میلیون ریال | ۵۰۰ ریال | تغییر مقیاس مقدار |

این مثال‌ها نشان می‌دهند که خطای یک نویسه می‌تواند مهم‌تر از خطای چندین کلمه باشد. جهت نمایش را با معنی مقدار اشتباه نگیرید. برای تشخیص، رشتهٔ خام، توالی Code Point، موقعیت تک‌تک کلمات و تصویر صفحه را کنار هم قرار دهید. اگر رشتهٔ خام درست و فقط UI غلط است، باید نمایش RTL و جداسازی جهت عبارت‌ها را اصلاح کنید؛ اگر رشتهٔ خام از ابتدا غلط است، تغییر CSS راه‌حل نیست.

### تشخیص عملی RTL؛ مشکل از متن است، ترتیب است یا نمایش؟

وقتی یک عبارت فارسی ـ انگلیسی یا یک عدد وارونه می‌بینید، قبل از انتخاب کتابخانهٔ اصلاح RTL این چک‌لیست را روی یک نمونهٔ کوچک اجرا کنید:

| مرحلهٔ تشخیص | چه چیزی بررسی شود؟ | تصمیم بعدی |
|---|---|---|
| کدهای واقعی متن | Code Point، PUA، U+FFFD، Presentation Forms، کنترل‌های جهت و نیم‌فاصله؛ مستقل از ظاهر مرورگر | اگر نگاشت فونت خراب است، Decoder یا OCR هدفمند؛ NFC/NFKC نگاشت گم‌شده را اختراع نمی‌کند |
| ترتیب حروف در کلمه | یک واژهٔ معلوم را با توالی منطقی مرجع، نه تصویر ظاهری، مقایسه کنید | اگر کلمه وارونه است، منبع Span و قرارداد خروجی موتور را بررسی کنید؛ Reverse عمومی ممنوع |
| توکن‌های ترکیبی | عدد، تاریخ، واحد پول، شناسه و عبارت لاتین را همراه نشانهٔ جهت و همسایگی‌شان بررسی کنید | عدد را به عنوان Run مستقل و با ترتیب داخلی معتبر نگه دارید؛ علامت یا صفر ابتدایی را خودسرانه جابه‌جا نکنید |
| ترتیب عناصر | کادرهای خط، ستون، پاراگراف و سلول را روی تصویر صفحه تطبیق دهید | اگر فقط ترتیب عناصر خراب است، Reading Order یا روابط جدول را اصلاح کنید، نه Recognizer |
| نمایش | همان رشتهٔ منطقی را در محیط استاندارد RTL با فونت فارسی و جداسازی جهت نمایش دهید | اگر رشتهٔ خام و روابط صحیح‌اند، مشکل UI است؛ نسخهٔ خام و نمایهٔ جست‌وجو را تغییر ندهید |

**کتابخانه‌ها کجا کمک می‌کنند؟** `unicodedata` برای مشاهده و نرمال‌سازی Unicode مناسب است؛ `python-bidi` (تابع `get_display`) رشتهٔ منطقی را برای نمایش بصری بازچینی می‌کند؛ `arabic-reshaper` شکل پیوستهٔ حروف را برای محیط رسم متنی که Shaping ندارد آماده می‌کند. در چنین محیط قدیمی‌ای ممکن است ترکیب Shaping و BiDi برای رسم لازم باشد. اما در مرورگر، Qt یا موتور دارای Shaping و BiDi استاندارد، غالباً همان متن منطقی و تنظیم جهت کافی است. خروجی Reshaper یا `get_display` را به‌عنوان متن مرجع یا متن Embedding ذخیره نکنید. هیچ‌کدام راه‌حل عمومی وارون‌سازی یک PDF بصری به منطقی نیستند. [۴۰](#ref-40)

برای Visual-to-Logical واقعی، از قرارداد Decoder/OCR، مختصات Glyph/Span، جهت Run، نگاشت فونت، بافت زبان و تصویر استفاده کنید. اگر دو بازسازی محتمل‌اند، هر دو را به‌عنوان Candidate نگه دارید و مورد حساس را برای بازبینی بفرستید. تست Regression باید خروجی خام، ترتیب منطقی بازسازی‌شده و تصویر نهایی را جدا مقایسه کند. مثلاً یک جمله با کد کالا `AB-0123`، تاریخ ۱۴۰۵/۰۸/۲۶، درصد و کلمهٔ فارسی را بررسی کنید: سالم بودن ارقام کافی نیست؛ باید بدانیم هرکدام متعلق به کدام فیلد هستند. برای تاریخ شمسی، تبدیل تقویم را به مرحلهٔ معنایی مستقل بسپارید و مقدار اصلی را حفظ کنید.

### چرا معکوس کردن خودکار عددها خطرناک است؟

در بعضی آزمایش‌های قبلی، مقدارهای عددی بعد از پردازش PDF فارسی با ترتیب اشتباه ظاهر می‌شدند. یک گزارش مرتبط با رفتار Backend راست‌به‌چپ Docling هم وجود داشت، اما علت دقیق باگ بالادستی در آن بررسی مستقلاً تأیید نشد. بنابراین بهتر است به مشاهدهٔ خطا تکیه کنیم، نه اینکه یک علت تأییدنشده را قطعی اعلام کنیم.

یکی از راهکارهایی که امتحان کردم، مقایسهٔ مقدار استخراج‌شده با متن Native بود: اگر Parser مقدار 21 را داده بود و منبع معتبر مقدار 12 را نشان می‌داد، می‌شد آن را کاندیدای اصلاح دانست. در تعدادی از نمونه‌ها این روش کمک کرد. اما اگر هر دو عدد روی صفحه وجود داشته باشند، یا خود لایهٔ Native خراب باشد، این قاعده قابل اتکا نیست. حتی وجود 12 در همان صفحه ثابت نمی‌کند که به همان سلول تعلق دارد.

بهتر است اصلاح را به **ناحیهٔ مشخص و شواهد معتبر همان ناحیه** محدود کنیم. اگر شواهد کافی نداریم، تعارض را ثبت کنیم و Crop را دوباره بخوانیم یا برای بازبینی بفرستیم. معکوس کردن همهٔ رقم‌ها با یک Rule سراسری فقط یک باگ را با باگ دیگری عوض می‌کند.

### ارقام، تاریخ و واحدهای فارسی

در سندهای واقعی ممکن است ارقام فارسی، عربی و لاتین با هم استفاده شوند. فاصله، جداکنندهٔ هزارگان و اعشار، علامت منفی، درصد، ممیز، تاریخ شمسی و شناسه‌ها هم باید با دقت بررسی شوند. مثلاً ۰۱۲۳ در یک شناسه لزوماً معادل 123 نیست. مبلغ ریال با تومان فرق دارد؛ تبدیل آن‌ها یک تصمیم دامنه‌ای است، نه وظیفهٔ خودکار OCR. تاریخ شمسی هم بدون دانستن تقویم و قالب ورودی نباید به تاریخ میلادی تبدیل شود.

برای فیلدهای عددی سه چیز را جدا نگه دارید: متن خام، مقدار تفسیرشده با نوع و واحد مشخص، و وضعیت اعتبارسنجی. اگر ابهام در جداکننده یا مقیاس وجود دارد، مقدار تفسیرشده می‌تواند نامعلوم بماند. برای محاسبات مالی از نوع دادهٔ اعشاری دقیق استفاده کنید و رقم‌ها، علامت‌ها و واحد را در تست‌های مستقل بررسی کنید.

### مطالعهٔ موردی: آیا مدل OCR تخصصی فارسی کافی است؟

**[آزمایش محدود]** Bina Rizeh روی ۳۷ Crop واقعی از چهار خانوادهٔ سند با معیارهای ثابت بررسی شد.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-14-title" direction="ltr" height="486" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 486" width="720" xml:lang="fa"><title id="v5-figure-14-title">نتیجهٔ خوانش Bina Rizeh</title><rect x="0" y="0" width="720" height="486" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="458" rx="8" width="720" x="0" y="0"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="360" y="32">نتیجهٔ خوانش Bina Rizeh</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="52"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="82">نرخ خطای نویسه (CER)</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="82">31.03%</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="116"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="146">نرخ خطای واژه (WER)</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="146">51.69%</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="180"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="210">تطابق کامل خط</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="210">9/37</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="244"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="274">علامت درصد</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="274">2/9</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="308"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="338">علامت جمع</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="338">3/9</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="372"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="402">علامت کوچک‌تر</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="402">2/7</text></g></svg><figcaption>[آزمایش محدود] روی ۳۷ برش واقعی؛ معیارهای حساس در متن توضیح داده شده‌اند.</figcaption></figure>

| معیار | نتیجهٔ ثبت‌شده |
|---|---:|
| Character Error Rate (CER) | ۳۱٫۰۳٪ |
| Word Error Rate (WER) | ۵۱٫۶۹٪ |
| تطابق کامل خط | ۹ از ۳۷ |
| تشخیص علامت درصد | ۲ از ۹ |
| تشخیص علامت جمع | ۳ از ۹ |
| تشخیص علامت کوچک‌تر | ۲ از ۷ |
| نمونه‌های حساس مردود طبق معیار آزمایش | ۲۱ از ۲۹ |

بعضی از موارد مردود به فاصله‌گذاری و مرزبندی متن مربوط بودند؛ این عدد به معنی ۲۱ خطای معنایی مالی نیست. نکتهٔ جالب‌تر این بود که در نمونه‌های منتخب، بعضی خطاهای علامت از همان خروجی خام مدل، قبل از اصلاح RTL، وجود داشتند. پس مسئله فقط ترتیب نمایش نبود و بخشی از مشکل در Recognition ایجاد شده بود.

این نتیجه باعث شد تخصصی بودن مدل برای زبان فارسی را به‌عنوان معیار کافی برای انتخاب OCR در اسناد مالی در نظر نگیرم. **این عددها متعلق به همان Snapshot هستند، نه حکم دربارهٔ کل خانوادهٔ Bina.** نسخه‌های جدیدتر و تنظیمات مختلف باید مستقلاً سنجیده شوند. کارت Bina-0.4-Rizeh، یک مسیر بازتولید عمومی برای ۱۹۳ Crop در تاریخ ۲۴ نوامبر ۲۰۲۶ ارائه می‌کند و تصریح می‌کند که آن مجموعه بارها در توسعه به کار رفته، پس تست دست‌نخورده نیست. می‌توان از آن برای Regression و یادگیری روش آزمایش استفاده کرد، ولی نمره‌های آن را با ۳۷ Crop من یا معیارهای دارای نرمال‌سازی و مخرج متفاوت مقایسهٔ مستقیم نکنید. [۴۲](#ref-42) ممکن است یک مدل فارسی روی نوع دیگری از متن عملکرد مناسبی داشته باشد. برای کاربرد خودمان باید فونت، کیفیت اسکن، اعداد، علامت‌های ریاضی و نوع سند را هم وارد ارزیابی کنیم.

اگر شما با چنین نتیجه‌ای مواجه شدید، قبل از Fine-tuning یا تعویض مدل این مراحل را انجام دهید:

1. تصویر اصلی و Crop واقعی ورودی را کنار هم بگذارید؛ آیا علامت یا نقطه هنوز دیده می‌شود؟
2. خروجی خام Recognizer را بدون هیچ اصلاحی ذخیره و با مرجع تطبیق دهید.
3. همان خروجی را بعد از نرمال‌سازی و بازسازی ترتیب مقایسه کنید؛ کدام خطاها تازه ایجاد شده‌اند؟
4. چند موتور دیگر را روی **همان Cropها**، با معیار و تنظیمات ثبت‌شده مقایسه کنید.
5. اگر خطای Recognition پابرجاست، آن‌وقت به تغییر مدل، دادهٔ آموزشی، Fine-tuning یا راه بازیابی فکر کنید.

یک مدل که ۹۰٪ متن معمولی را درست می‌خواند ممکن است درصد، منفی یا رقم اعشار را بیشتر از حد قابل قبول حذف کند. برای کاربرد حساس، معیارها باید متناسب با اثر خطا باشند.

### چرا برای اسکن فارسی VLM-first را ترجیح می‌دهم؟

Bina از گزینه‌های تخصصی فارسی مورد توجه من است، اما نتیجهٔ ۳۷ Crop بالا برای فیلدهای حساس کافی نبود. فعلاً روی اسکن یا عکس فارسی که قرار است مبنای تصمیم مالی یا سازمانی باشد، OCR متعارف را مسیر اصلی پذیرش قرار نمی‌دهم. برای خوانش متن از پیکسل سراغ یک VLM دستورپذیر مناسب می‌روم؛ Gemini Flash یکی از اولین گزینه‌هایی است که آزمایش می‌کنم. مدل‌های GPT، Claude، Qwen و Gemma هم بسته به محدودیت داده، هزینه و کیفیت، کاندیدای مقایسه‌اند. این انتخاب مبتنی بر تجربه و شواهد محدود است؛ هنوز ادعای برنده بودن همهٔ VLMها بر همهٔ OCRها را نداریم.

دو گزارش بیرونی نیز جهت این فرضیه را تقویت می‌کنند. در یک مقایسهٔ اکتشافی با تصاویر ساخته‌شده توسط نویسنده، سه مدل روی ورودی‌های یکسان ارزیابی شده‌اند [۴۷](#ref-47):

| مدل | CER متن فارسی | CER متن فارسی و لاتین ترکیبی |
|---|---:|---:|
| Bina 0.2 Rizeh | ۴٫۸۸٪ | ۳۹٫۲۳٪ |
| Gemma 4 31B | ۰٫۷۴٪ | ۰٫۶۱٪ |
| Qwen3.8 27B | ۴٫۳۲٪ | ۰٫۷۲٪ |

ده تصویر فارسی خالص در آن گزارش از پنج متن ساخته‌شده در دو وضعیت کیفیت تشکیل شده‌اند؛ پنج تصویر ترکیبی نیز کنترل مصنوعی هستند. صفحات طبیعی بدون متن مرجع در نمره‌های بالا دخیل نیستند. خروجی‌های خام و ابزار امتیازدهی منتشر شده‌اند، اما سخت‌افزار یکسان نبوده و این مقایسه، Benchmark سرعت یا نمایندهٔ همهٔ اسناد واقعی فارسی نیست.

گزارش مقدماتی دیگری روی ۲۱ تصویر دست‌خط فارسی که هر سه موتور پردازش کرده‌اند، CER حدود ۸٫۳۰٪ برای Gemini 2.5 Flash، ۸۱٫۹۹٪ برای RapidOCR با مدل عربی و ۹۰٫۶۸٪ برای Tesseract با زبان فارسی گزارش می‌کند. اجرای آزمایش کامل نشده بود و مدل‌های دیگر هنوز اجرا نشده بودند؛ این صرفاً یک مقایسهٔ محدود است. [۴۸](#ref-48)

نکتهٔ آموزشی این است که موفقیت روی متن فارسی خالص را به ترکیب فارسی و انگلیسی یا دست‌خط و جدول تعمیم ندهید. VLM هم ممکن است رقم را حدس بزند، متن را به شکل آشناتر اصلاح کند یا سلول اضافی بسازد. بنابراین VLM-first به معنی VLM-only یا پذیرش بدون بررسی نیست. متن دیجیتال سالم و هندسه را نگه دارید، خوانش بصری را تا جای ممکن به ناحیهٔ لازم محدود کنید، و برای مقادیر حساس از شاهد مستقل و بازبینی استفاده کنید. اگر موتور سبک روی یک قالب محدود و دادهٔ واقعی به حد پذیرش می‌رسد، دلیلی ندارد صرفاً به‌خاطر اندازهٔ مدل آن را کنار بگذارید.

### فارسی در Layout و جدول

چالش فارسی در سطح نویسه تمام نمی‌شود. در سندهای سازمانی راست‌به‌چپ، ترتیب ستون‌ها، شماره‌گذاری بندها، جدول‌های عمودی ادغام‌شده، تیترهای ترکیبی و پاورقی‌ها هم ممکن است از فرض‌های یک Parser آموزش‌دیده بر سندهای لاتین متفاوت باشند. برای جدول، جهت نمایش را از شناسهٔ منطقی ستون جدا کنید. ستون شمارهٔ صفر صرفاً به دلیل RTL نباید به ستون آخر تبدیل شود؛ قرارداد اندیس‌گذاری، ارتباط سرستون و هندسه باید مشخص باشند.

برای سند حقوقی فارسی، «مادهٔ ۳۷» و «مادهٔ ۳۷-۲» را یک عنوان واحد فرض نکنید. برای جدول، واحد بالای جدول یا ستونی که چند ردیف را پوشش می‌دهد باید به همهٔ مقدارهای مربوط متصل باشد. در طراحی Gold Dataset فارسی، این موارد را به‌عنوان خانواده‌های جدا تعریف کنید، نه صرفاً چند صفحهٔ متن با فونت فارسی.

### دامنهٔ پشتیبانی فارسی در MVP

در محصولی که فقط PDF تایپی سالم را پشتیبانی می‌کند، Router باید اسکن و لایهٔ متن خراب را تشخیص دهد و وضعیت Unsupported یا Review Required بدهد؛ خروجی خالی را موفق ثبت نکند. اگر اسکن جزو نیاز اصلی مشتریان است، کیفیت Recognition و بازسازی ساختار را روی نمونه‌های فارسی همان مشتری بسنجید؛ معیارهای بخش ۱۳ مبنای مقایسه‌اند.

### یک نگاه به تجربهٔ توسعه‌دهندگان فارسی و عربی خارج از گروه

در یک پرسش قدیمی Stack Overflow دربارهٔ PDF فارسی و عربی، کاربر از معکوس شدن واژه‌ها و نویسه‌ها و مشکل نیم‌فاصله شکایت داشت. در بحثی دیگر در r/learnpython، اعداد فارسی و حتی تغییر یک رقم مسئلهٔ اصلی گزارش شده بود. در پروژهٔ Docling چند گزارش مستقل ارزش مطالعه دارند: Issue شمارهٔ ۱۹۳۸ دربارهٔ وارونگی کلمه و حروف، ۱۴۲۱ و ۲۱۷۹ دربارهٔ پشتیبانی RTL عربی، ۳۰۲۱ دربارهٔ محدودیت پیکربندی OCR اسکن عربی، و ۳۴۶۲ دربارهٔ خروجی خراب PDF ترکیبی عربی ـ انگلیسی. این‌ها گزارش‌های نسخه و نمونهٔ مشخص‌اند؛ وضعیت بسته یا باز بودن Issueها به تنهایی اثبات کیفیت نسخهٔ فعلی نیست. اما تکرار همین کلاس خطاها نشان می‌دهد باید پشتیبانی اسکریپت عربی را در ترکیب PDF Backend، OCR، بازسازی ترتیب و صادرکنندهٔ متن بسنجیم، نه از روی فهرست زبان‌های مدل. برای فارسی، فونت‌ها، اسناد ترکیبی و جدول‌های خودتان را به Regression Set اضافه کنید. [۴۱](#ref-41) راهکار نمایش راست‌به‌چپ را هم نباید بدون بررسی به اصلاح رشتهٔ خام تبدیل کرد. [۳۲](#ref-32)

در یک تجربهٔ r/LocalLLaMA دربارهٔ فرم‌های عربی، نویسنده از مهرهای روی متن، سلول‌های ادغام‌شده و دست‌خط به‌عنوان عوامل سختی نام برد و کیفیت خوب Gemini در نمونه‌های خودش را همراه با محدودیت اجرای محلی و مکان‌یابی کلمات مطرح کرد. این مشاهده با تجربهٔ گروه ما هم‌جهت است، اما عربی و فارسی یک مجموعهٔ ارزیابی واحد نیستند. نه اعداد آن پست، نه تعداد نمونه‌های پیشنهادی برای Fine-tuning، نه ادعای برتری یک مدل را نباید بدون دادهٔ مرجع و مجوز بازتولیدپذیر به فارسی تعمیم داد. [۳۳](#ref-33)

برای یک محصول فارسی، «چندزبانه» بودن مدل را با پنج قابلیت مجزا امتحان کنید: خواندن حروف فارسی، خواندن عدد و علامت، ترکیب فارسی و انگلیسی، ترتیب صفحه/جدول و دست‌خط یا تصاویر کم‌کیفیت. در مرحلهٔ ارزیابی به خانوادهٔ سند هم دقت کنید: نتیجهٔ خوب روی کتاب تایپی تضمین نمی‌کند همان مدل روی فاکتور، فرمول یا بخشنامه درست کار کند. یک مسیر بومی ممکن است از Native + OCR سبک + VLM محلی ساخته شود، و یک مسیر API از همان Native + یک سرویس بصری خارجی؛ تفاوت را باید در معیارهای کاربرد، محرمانگی و هزینه اندازه گرفت.

## ۹. DocumentIR و TableIR؛ خروجی Markdown را با خود سند اشتباه نگیریم

Markdown برای خواندن انسان و بسیاری از مدل‌های زبانی راحت است. اما جدول چندسطحی، سلول ادغام‌شده، چند Caption یا Provenance دقیق ممکن است در Markdown ساده از دست بروند. اگر فقط Markdown نگه دارید، اصلاح یک خطای جدول در آینده دشوار می‌شود.

Intermediate Representation یا IR یعنی نمایش میانی داده. اسم‌های DocumentIR و TableIR در این مقاله نام‌های مفهومی‌اند؛ استاندارد اجباری یا نام یک API عمومی نیستند. می‌توانید از ساختارهایی مثل DoclingDocument یا Schema مناسب نیاز خودتان استفاده کنید. [۳](#ref-3)

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-15-title" direction="ltr" height="263" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 263" width="720" xml:lang="fa"><title id="v5-figure-15-title">تمام نمایش‌های مشتق‌شده از یک ساختار canonical تولید می‌شوند.</title><defs><marker id="arrow-15" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="263" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#f6f5f0" height="45" rx="8" stroke="#aab8b5" width="350" x="185" y="8"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="36">سند اصلی</text><path d="M360 53L360 78" fill="none" marker-end="url(#arrow-15)" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="56" rx="8" stroke="#176e73" width="430" x="145" y="78"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="101">DocumentIR / TableIR</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="121">نمایش مرجع ساختار</text><path d="M360 134V157H90V180" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="40" rx="8" stroke="#aab8b5" width="160" x="10" y="180"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="90" y="206">JSON</text><path d="M360 134V157H270V180" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="40" rx="8" stroke="#aab8b5" width="160" x="190" y="180"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="270" y="206">HTML</text><path d="M360 134V157H450V180" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="40" rx="8" stroke="#aab8b5" width="160" x="370" y="180"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="450" y="206">Markdown</text><path d="M360 134V157H630V180" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="40" rx="8" stroke="#aab8b5" width="160" x="550" y="180"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="630" y="206">قطعه‌ها</text></g></svg><figcaption>تمام نمایش‌های مشتق‌شده از یک ساختار canonical تولید می‌شوند.</figcaption></figure>

DocumentIR می‌تواند سلسله‌مراتب سند، صفحه‌ها، عنوان‌ها، پاراگراف‌ها، شکل‌ها، جدول‌ها، فهرست‌ها، ترتیب خواندن و پیوند به شواهد را نگه دارد. TableIR برای حفظ ساختار دقیق یک جدول مناسب است: سلول، سطر، ستون، محدودهٔ ادغام، عنوان و سرستون، واحد، پاورقی، ادامهٔ جدول و وضعیت هر بخش.

به‌جای مثال کدنویسی، یک سلول فرضی را در نظر بگیرید که مقدار ۱۸٪ دارد. این اطلاعات باید دربارهٔ آن قابل بیان باشد:

| دسته | نمونهٔ اطلاعات |
|---|---|
| هویت | شناسهٔ سند، جدول و سلول |
| موقعیت | صفحهٔ فیزیکی، صفحهٔ چاپی، BBox، سیستم مختصات |
| ساختار | سطر و ستون منطقی، Rowspan/Colspan، ارتباط با سرستون |
| متن | رشتهٔ خام «۱۸٪»، متن نرمال‌شده و در صورت اعتبار، مقدار اعشاری و واحد |
| منشأ | Native، OCR یا VLM؛ نسخهٔ ابزار، تنظیمات، Artifact مرجع |
| کیفیت | کاندیدا، نیازمند بررسی، تعارض، پذیرفته‌شده یا قرنطینه |

اگر جدول اصلاح شد، خروجی HTML، Markdown و Chunkها باید از همان نسخهٔ پذیرفته‌شده دوباره ساخته شوند. در غیر این صورت ممکن است JSON جدید درست باشد ولی Vector Database هنوز مقدار قبلی را به مدل زبانی بدهد.

فرمت Canonical یک منبع حقیقت عملیاتی است، نه ادعای اینکه خودبه‌خود درست است. فایل اصلی و شواهد خام همچنان باید در دسترس باشند. و اگر بخشی از سند نیازمند بازبینی است، این وضعیت نباید هنگام تبدیل به Markdown ناپدید شود.

## ۱۰. Vision-Language Models؛ خوانش دستورپذیر و استخراج ساختاریافته

VLM مدلی است که می‌تواند ورودی تصویری و زبانی را پردازش کند. در Parsing می‌توانیم از آن برای خواندن یک صفحه، یک جدول یا Crop دشوار استفاده کنیم. اما نوع وظیفه را روشن کنید: **رونویسی دقیق از تصویر** با **توصیف تصویر**، **تفسیر سند** و **تولید ساختار جدید** یکسان نیست.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-16-title" direction="ltr" height="278" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 278" width="720" xml:lang="fa"><title id="v5-figure-16-title">فقط ناحیه مشکوک و مشخص برای رونویسی تصویری ارسال می‌شود.</title><defs><marker id="arrow-16" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="278" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="230" rx="8" stroke="#aab8b5" width="190" x="35" y="10"/><rect fill="#5f6964" height="9" rx="2" stroke="none" width="93.1" x="111.0" y="28"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="50" y="52"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="50" y="64"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="50" y="76"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="50" y="88"/><rect fill="#fffdf9" height="70" rx="2" stroke="#aab8b5" width="160" x="50" y="110"/><rect fill="#e1f1ef" height="17.5" rx="1" stroke="none" width="160" x="50" y="110"/><path d="M50 127.5H210" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M50 145.0H210" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M50 162.5H210" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M103.33333333333334 110V180" fill="none" stroke="#aab8b5" stroke-width="1"/><path d="M156.66666666666669 110V180" fill="none" stroke="#aab8b5" stroke-width="1"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="160" x="50" y="192"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="148" x="62" y="202"/><rect fill="#aab8b5" height="4" rx="1" stroke="none" width="136" x="74" y="212"/><rect fill="none" height="34" rx="2" stroke="#94600d" width="160" x="50" y="108"/><path d="M225 126L290 126" fill="none" marker-end="url(#arrow-16)" stroke="#176e73" stroke-width="2"/><rect fill="#fff1d9" height="64" rx="8" stroke="#94600d" width="180" x="295" y="94"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="385" y="117">برش ناحیه</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="385" y="140">نرخ ۱۸٪</text><path d="M475 126L530 126" fill="none" marker-end="url(#arrow-16)" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="64" rx="8" stroke="#176e73" width="150" x="535" y="94"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="22" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="610" y="132">VLM</text></g></svg><figcaption>فقط ناحیه مشکوک و مشخص برای رونویسی تصویری ارسال می‌شود.</figcaption></figure>

اگر لایهٔ Native جدول خراب است ولی هندسهٔ آن سالم است، می‌توانیم ناحیهٔ مورد نظر را با شناسهٔ ثابت جدا کنیم و از مدل بخواهیم فقط متن همان ناحیه را بخواند. اگر ساختار جدول هم خراب است، باید مسئلهٔ بازسازی را جدا تعریف کنیم. در هر دو حالت، خروجی VLM یک کاندیداست و باید با شواهد بررسی شود.

### تفاوت بنیادی: Prompt و Structured Output

یکی از دلایلی که VLMهای دستورپذیر را برای محصولات واقعی ترجیح می‌دهم این است که فقط موتور تصویر به متن نیستند. برای یک OCR متعارف، معمولاً باید زبان، مدل Recognition و تنظیمات را مشخص کنید، بعد متن و کادرها را با کد، Regex، قواعد قالب یا یک مدل دیگر به فیلدهای محصول تبدیل کنید. ولی در VLM می‌توانید همان تصویر یا PDF را همراه یک دستور زبان طبیعی بدهید و از مدل بخواهید دقیقاً وظیفهٔ تعریف‌شده را انجام دهد. در مدل و API سازگار، می‌توانید علاوه بر دستور، JSON Schema هم بدهید تا شکل خروجی به قرارداد نرم‌افزار نزدیک‌تر باشد. [۴۹](#ref-49)

**[مثال فرضی]** یک فاکتور فارسی با چند ردیف کالا، تخفیف و مبلغ نهایی را در نظر بگیرید. برای آرشیو، دستور شما می‌تواند رونویسی تمام متن باشد؛ برای سیستم حسابداری، همان ورودی را با دستور استخراج شمارهٔ فاکتور، تاریخ، واحد پول و ردیف‌های کالا به مدل بدهید. برای بررسی خطا، فقط Crop یک سلول مشکوک را بفرستید و بخواهید متن دقیق آن را بدون اصلاح گزارش کند. موتور را عوض نکرده‌اید؛ وظیفه و قرارداد خروجی را تغییر داده‌اید.

| قابلیت | OCR متعارف به‌عنوان موتور Recognition | VLM دستورپذیر |
|---|---|---|
| تعیین وظیفه | تنظیمات زبان، Detection، Recognition و قطعه‌بندی | Prompt برای رونویسی، استخراج فیلد، تشخیص رابطه یا تفسیر |
| تغییر قالب فاکتور یا فرم | ممکن است به قواعد جدید یا آموزش/تنظیم نیاز باشد | می‌توان دستور و Schema را تغییر داد؛ ارزیابی دوباره لازم است |
| خروجی | متن، کادر و Confidence در موتورهای سازگار | متن، Markdown، HTML یا JSON؛ Structured Output در مدل/API سازگار |
| اطلاعات ناخوانا | Confidence یا خروجی خالی، بسته به موتور | می‌توان صریحاً درخواست مقدار نامعلوم و دلیل کرد؛ تضمین تبعیت معنایی نیست |
| اتصال به محصول | معمولاً مرحلهٔ نگاشت و بازسازی جدا لازم است | ممکن است چند مرحله در یک درخواست ادغام شوند؛ Validation همچنان لازم است |

برای نمونه، یک دستور رونویسی مناسب می‌تواند چنین باشد: «متن این ناحیه را عیناً و در ترتیب منطقی فارسی بخوان؛ ارقام و علامت‌ها را تغییر نده؛ چیزی را ترجمه، خلاصه یا اصلاح نکن؛ اگر بخشی ناخواناست آن را نامعلوم گزارش کن؛ مقدار را به شناسهٔ ناحیه‌ای که دریافت کرده‌ای وصل کن.» دستور استخراج فیلد برای همان ناحیه می‌تواند صرفاً شمارهٔ فاکتور، تاریخ خام، مبلغ خام، واحد و شناسهٔ شاهد را بخواهد. فیلد نرمال‌شده و مقدار محاسبه‌شده را از متن خام جدا نگه دارید.

Structured Output با JSON معمولی فرق دارد: در APIهایی که از Schema پشتیبانی می‌کنند، سازوکار تولید می‌تواند ساختار و نوع داده را محدود کند؛ صرفاً درخواست «JSON بده» چنین تضمینی ندارد. پشتیبانی از Schema و محدودیت‌های آن بین مدل‌ها و Endpointها فرق دارد. حتی با خروجی مطابق Schema، مقدار اشتباه یا ردیف خیالی می‌تواند از نظر نحوی معتبر باشد. برای نمونه اگر واحد ریال به تومان تغییر کند، JSON همچنان معتبر است. بنابراین دستور و Schema را نسخه‌بندی کنید و قواعد معنایی و شواهد را بیرون مدل اعتبارسنجی کنید. در پردازش سند غیرقابل‌اعتماد، دستورهای داخل سند را داده بدانید، نه دستور سیستم؛ مدل نباید مجاز به اجرای عملیات بر اساس متن سند باشد.

### چرا در محصولات سازمانی حتی PDF دیجیتال را به VLM می‌دهند؟

در پروژه‌های واقعی، هدف شرکت معمولاً «متن OCR» نیست؛ می‌خواهد فاکتور وارد ERP شود، قرارداد به فیلدهای قابل جست‌وجو تبدیل شود یا سند در RAG به شاهد معتبر تبدیل شود. برای سندهای متنوع، یک VLM دستورپذیر با خروجی ساختاریافته می‌تواند زمان توسعهٔ قواعد مخصوص هر قالب، تعداد مؤلفه‌های لازم و اصطکاک تغییر Schema را کاهش دهد. حتی روی PDF دیجیتال، خواندن هم‌زمان متن و ظاهر صفحه برای ارتباط سرستون و مقدار، فرم، نمودار یا متن پراکنده مزیت عملی دارد. در تجربهٔ خودم نیز برای اسناد فارسی دارای ساختار پیچیده، گاهی Gemini را حتی وقتی OCR الزامی نبود ترجیح دادم. این انتخاب دربارهٔ آسان‌تر شدن ساخت محصول و کیفیت مورد انتظار آن است، نه اینکه متن دیجیتال سالم را همیشه دوباره تولید کنیم.

گزارش‌های صنعت هم حرکت از OCR تخت به Document AI ساختاریافته و استدلال چندوجهی انتخابی را نشان می‌دهند. نظرسنجی AIIM و Deep Analysis در سال ۲۰۲۵ از ۶۰۰ شرکت در چهار کشور گزارش کرده که ۷۸٪ از پاسخ‌دهندگان از فناوری AI به‌صورت عملیاتی استفاده می‌کردند و ۶۶٪ پروژه‌های جدید IDP برنامهٔ جایگزینی سیستم موجود داشتند. این اعداد دربارهٔ پذیرش عمومی AI و تغییر سیستم‌اند، نه سهم VLM از موتورهای استخراج. گزارش Verdantix در مارس ۲۰۲۶ نیز معماری چندمرحله‌ای، Layout-first و استدلال چندوجهی انتخابی را از مسیرهای مهم Document AI سازمانی می‌داند. [۵۰](#ref-50)

نمونهٔ صنعتی Justt از ترکیب OCR و مؤلفهٔ استخراج مبتنی بر AI برای اسناد Chargeback استفاده می‌کند و راهنمای Gemini Layout Parser در Google Document AI هم ترکیب مدل‌های OCR تخصصی و Gemini را توضیح می‌دهد. [۵۱](#ref-51) در نتیجه شواهد برای حرکت بازار به سمت VLM و Document AI چندوجهی وجود دارد، اما دادهٔ نماینده‌ای ندارم که نشان دهد اکثریت شرکت‌ها OCR را به‌طور کامل کنار گذاشته‌اند. معماری ترکیبی هم انتخاب جدی سازمانی است.

از نظر من برای یک محصول فارسی با فرم‌ها و سندهای متنوع، VLM مستقیم یا مسیر VLM-first یکی از بهترین نقطه‌های شروع است، حتی اگر بخشی از سند دیجیتال باشد. سه مسیر را بر اساس خروجی یکسان مقایسه کنید: Native + قواعد استخراج، VLM مستقیم با PDF و Schema، و Native + VLM هدفمند. هزینهٔ توسعه، نگهداری، پوشش فیلد، پذیرش اشتباه، محرمانگی، p95 و هزینهٔ هر رکورد پذیرفته‌شده را اندازه بگیرید. اگر مسیر Native ساده همان کیفیت را با هزینهٔ کمتر می‌دهد، آن را حفظ کنید؛ اگر VLM با وجود هزینهٔ بیشتر، خطا و بازبینی و تغییر قواعد را کم می‌کند، ممکن است از نظر کل هزینهٔ محصول اقتصادی‌تر باشد. Reliability را با تست و Quality Gate بسنجید، نه تعداد کمتر سرویس‌ها یا شهرت مدل.

### چه زمانی از Gemini Flash، Claude Opus، GPT یا Qwen استفاده کنیم؟

مدل را برای وظیفهٔ مشخص انتخاب کنید: رونویسی، تشخیص رابطهٔ بصری یا تفسیر. استدلال حقوقی قوی لزوماً به معنی رونویسی دقیق ارقام فاکتور نیست.

| خانواده یا مسیر | نقش منطقی برای آزمایش | چه زمانی مناسب نیست یا باید احتیاط کرد؟ |
|---|---|---|
| Gemini Flash / Flash-Lite دارای ورودی تصویر و PDF | Baseline کم‌اصطکاک و کاندیدای اصلی خوانش اسکن فارسی، استخراج فیلدهای متنوع و پردازش دسته‌ای در صورت مناسب بودن هزینهٔ واقعی | بازنویسی کل متن سالم، اسناد محرمانه بدون مجوز، ادعای پوشش کامل یا صحت عدد بدون بازبینی |
| Gemini Pro یا Claude Opus دارای Vision | بررسی مورد دشوار که مدل سریع‌تر در فهم Layout، نمودار یا رابطهٔ پیچیده شکست خورده؛ تحلیل و تفسیر سند در صورت نیاز | استفادهٔ پیش‌فرض از مدل گران‌تر برای رونویسی ساده؛ هزینه و محدودیت ورودی/خروجی باید اندازه‌گیری شود |
| مدل‌های GPT با ورودی تصویر/PDF | رونویسی، استخراج فیلد مبتنی بر Schema، تحلیل بصری و پاسخ مستند؛ Baseline دیگر برای مقایسه | فرض اینکه Structured Outputs صحت محتوا را تضمین می‌کند یا موقعیت تولیدی دقیقاً مختصات PDF است |
| Qwen-VL و سایر VLMهای وزن‌باز مناسب Vision | آزمایش مسیر محلی، خصوصاً وقتی کنترل داده و نسخه یا تعویض مدل مهم است؛ Crop OCR و فهم Layout | صرفاً به دلیل وزن‌باز بودن کیفیت یا هزینهٔ کمتر را فرض نکنید؛ VRAM، رزولوشن، Quantization، مجوز و توان زبان را بسنجید |
| VLM تخصصی سند مانند Granite-Docling، dots.ocr، olmOCR یا مدل‌های خانوادهٔ PaddleOCR-VL | استخراج گستردهٔ متن و ساختار با قرارداد تخصصی؛ مقایسه با VLM عمومی و Pipeline کلاسیک | وقتی هدف فقط تفسیر یک سؤال است شاید بیش از نیاز باشد؛ خروجی، پوشش و فارسی را مستقل آزمایش کنید |
| مدل زبانی بدون Vision | نگاشت متنِ سالم به Schema، طبقه‌بندی یا تحلیل روی دادهٔ ساختاریافته | برای متن ناخوانا داخل تصویر، اطلاعاتی که Parser از دست داده یا تشخیص هندسهٔ واقعی کافی نیست |

Gemini تجربهٔ ثبت‌شدهٔ نویسنده است؛ Claude Opus، GPT و Qwen گزینه‌های مقایسه‌اند. شناسه و ورودی واقعاً پشتیبانی‌شدهٔ مدل/API را ثبت کنید. BBox تولیدی مدل محلی را به مختصات صفحه نگاشت و بررسی کنید. [۳۰](#ref-30) [۳۴](#ref-34)

### انتخاب مدل بر اساس نوع خطا

مدل قوی‌تر را فقط روی مواردی آزمایش کنید که مسیر پایه در آن‌ها شکست خورده است. اگر خطا از Crop، نگاشت BBox یا جابه‌جایی سرستون است، ابتدا همان مرحله را اصلاح کنید. در رونویسی، خروجی باید متن واقعی را حفظ کند؛ در استخراج فیلد، مقدار نامعلوم و شاهد را گزارش کند؛ در تفسیر نمودار، استنتاج را از مقدار عیناً نوشته‌شده جدا کند. برای هزینه و مقایسهٔ مدل‌های عمومی و تخصصی به بخش ۱۳ مراجعه کنید.

### Gemini API؛ خوانش اصلی اسکن فارسی و Fallback هدفمند برای PDF دیجیتال

در طرح Adaptive PDF Extraction که در ۱۱ اکتبر ۲۰۲۶ در کانال توضیح دادم، ایدهٔ اصلی استفاده از PyMuPDF برای متن و مختصات، Docling برای بازسازی ساختار و Gemini برای خوانش دشوار بود. در بعضی آزمایش‌ها ترکیب Docling و Gemini نتیجهٔ قابل استفاده‌ای برای نمونهٔ موردنظر داد، اما آن تجربه را نباید پذیرش همهٔ اسناد دانست؛ اندازه‌گیری‌های بعدی همین مقاله محدودیت‌ها را نشان می‌دهند.

Gemini API ورودی PDF دارد [۱۷](#ref-17). برای ناحیهٔ ناخوانا، ارسال کل صفحه را با Crop و شناسهٔ ناحیه مقایسه کنید؛ اگر سرستون یا واحد خارج Crop است، شاهد پیرامونی را هم بفرستید. در APIهای سازگار، تنظیم `media_resolution` یا `resolution` بودجهٔ توکن ورودی بصری و در نتیجه جزئیات قابل پردازش، هزینه و زمان را تغییر می‌دهد؛ DPI رندر محلی و رزولوشن داخلی مدل یک چیز نیستند. نام پارامتر، سطوح مجاز، مقدار پیش‌فرض و Token Accounting به مدل و Endpoint بستگی دارد. برای مثال، راهنمای فعلی Interactions API در حالت سندِ مستندشده از سطح low/medium/high و بودجه‌های تقریبی ۲۸۰/۵۶۰/۱٬۱۲۰ توکن تصویری برای هر صفحه در مدل‌های Gemini 3 و 4 نام می‌برد؛ این اعداد را به همهٔ مسیرهای API، لایهٔ Native یا قیمت نهایی تعمیم ندهید. [۴۳](#ref-43) در آزمایش Crop و کل صفحه، رزولوشن، توکن مصرف‌شده، کیفیت فیلد، هزینهٔ بازبینی و پوشش را با هم ثبت کنید.

درخواست مدل را به رونویسی دقیق همان ناحیه محدود کنید: بدون ترجمه، بدون خلاصه، بدون اصلاح املایی و بدون حدس برای مقدار ناخوانا. خروجی باید بتواند مقدار نامعلوم و تعارض را گزارش کند. این دستورها احتمال خطا را کم می‌کنند، ولی ضمانت صحت نیستند؛ Schema فقط شکل خروجی را کنترل می‌کند. استفاده از مدل‌های کم‌هزینه‌تر خانوادهٔ Flash برای Baseline ارزش آزمایش دارد، اما هزینهٔ واقعی شامل تعداد صفحه، Retry، توکن خروجی، بازبینی و پردازش دوباره است. اینکه سند در RAG معمولاً هنگام Ingest خوانده می‌شود، به معنی هزینهٔ صفر در به‌روزرسانی، تغییر Parser یا هر پرس‌وجو نیست.

### مطالعهٔ موردی: Gemini 3 Flash و Structured Output

**[آزمایش محدود]** Gemini 3 Flash برای خواندن دادهٔ جدولی استفاده شد:

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-18-title" direction="ltr" height="294" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 294" width="720" xml:lang="fa"><title id="v5-figure-18-title">نتیجهٔ استخراج جدولی Gemini 3 Flash</title><rect x="0" y="0" width="720" height="294" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="266" rx="8" width="720" x="0" y="0"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="360" y="32">نتیجهٔ استخراج جدولی Gemini 3 Flash</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="52"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="82">تطابق دقیق سلول</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="82">17/26</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="116"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="146">سلول اضافه</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="146">13</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="180"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" text-anchor="middle" unicode-bidi="isolate" x="510" y="210">رخداد بحرانی ازدست‌رفته</text><text direction="ltr" fill="#176e73" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="160" y="210">1/49</text></g></svg><figcaption>[آزمایش محدود] سه معیار مستقل، نه نمرهٔ واحد.</figcaption></figure>

| معیار | نتیجه |
|---|---:|
| متن سلول‌های کاملاً مطابق مرجع | ۱۷ از ۲۶ (حدود ۶۵٫۴٪) |
| سلول‌های اضافی تولیدشده | ۱۳ |
| رخداد عددی یا علامت بحرانی ازدست‌رفته | ۱ از ۴۹ |

این معیارها چیزهای متفاوتی را می‌سنجند. ۱۷ از ۲۶ دربارهٔ متن سلول‌های مرجع است و عدد ۱۳ دربارهٔ سلول‌های اضافی؛ ۱ از ۴۹ نیز رخدادهای بحرانیِ از دست رفته را گزارش می‌کند. نباید از یکی از آن‌ها به‌تنهایی نتیجه بگیریم که ساختار کل جدول درست بوده است.

**درس اصلی:** Structured Output تضمین نحویِ پیروی از Schema را هدف می‌گیرد؛ اعتبار معنایی داده مسئله‌ای جداست. حتی وقتی پاسخ JSON کاملاً معتبر باشد، ممکن است مدل سلولی بسازد که در سند وجود ندارد، عددی را اشتباه بخواند یا ارتباط سطر و ستون را تغییر دهد.

برای حل این مسئله، قرارداد خروجی را به شناسه‌های ناحیه محدود کنید، وجود هر رکورد را در فهرست مورد انتظار بررسی کنید، رکورد اضافی و تکراری را رد یا قرنطینه کنید و محتوای حساس را به تصویر و هندسه متصل کنید. اگر خوانش ناممکن بود، نبود پاسخ را صریح ثبت کنید؛ مدل نباید برای کامل به نظر رسیدن جدول حدس بزند.

### PageIndex Page Extract و استخراج هدایت‌شده

**[آزمایش محدود]** PageIndex Page Extract در یک آزمایش هدایت‌شده با Schema چند سطر و یک فرمول را استخراج کرد؛ نشانه‌ای برای بررسی این مسیر در نمونه‌های دشوار. Page Extract یک مسیر استخراج مبتنی بر Schema است؛ به‌خودی‌خود قرارداد بازسازی کامل سند، Reading Order و پوشش همهٔ عناصر را اثبات نمی‌کند. [۲۲](#ref-22)

### Inkling 2 Mini؛ موفقیت متنی با توانایی بینایی یکی نیست

**[آزمایش محدود؛ اطلاعات بازتولید ناقص]** در غربال اولیهٔ بینایی روی یک تصویر مالی مصنوعی، این مدل فقط سه مورد از پنج رقم بحرانی و هیچ‌کدام از دو عبارت مالی کامل را درست استخراج کرد. Inkling 2 Mini در ۲۰ نوامبر ۲۰۲۶ منتشر شد و طبق کارت مدل، معماری Transformer حلقه‌ای با لایه‌های MoE تنک دارد. [۳۹](#ref-39) نسخهٔ دقیق Endpoint/Checkpoint، Runtime، Quantization و Reasoning Effort آزمایش در گزارش در دسترس ثبت نشده‌اند؛ نمی‌توانم تعیین کنم کدام تنظیمات یا حتی چه مسیر Serving استفاده شده است. این مشاهده فقط برای همان غربال و همان خروجی معتبر است، نه نتیجهٔ منصفانهٔ مقایسهٔ مدل. برای تکرار، شناسه و Revision مدل، نسخهٔ Runtime و مسیر پردازش تصویر، دقت وزن‌ها/Quantization (در صورت کاربرد)، Reasoning Effort، محدودیت توکن، Prompt و Crop را ثبت کنید؛ اگر API این اطلاعات را افشا نمی‌کند «نامشخص» بنویسید. تازگی انتشار و پشتیبانی Runtime انگیزهٔ آزمایش دوباره است، نه توضیح اثبات‌شدهٔ علت شکست.

وقتی مدلی را برای OCR یا Document Understanding انتخاب می‌کنید، اول قرارداد ورودی تصویر و توانایی انجام یک وظیفهٔ کوچک را آزمایش کنید. امتیاز Reasoning یا کیفیت جواب‌های متنی جایگزین این غربال نیست.

### Confidence Score و Multi-Engine Consensus

اگر OCR برای یک عدد Confidence بالایی گزارش کند، آیا عدد را بپذیریم؟ نه صرفاً به خاطر این نمره. در یکی از آزمایش‌های ثبت‌شده، حتی خوانش‌هایی با Confidence بالا دارای خطا در عدد یا علامت مهم بودند. ابتدا باید بدانیم Confidence در آن موتور چه معنایی دارد و آیا روی نمونه‌های مرجع ما کالیبره شده است. احتمال یا امتیاز خام دو مدل مختلف را مستقیم با هم مقایسه نکنید.

فرض کنید برای یک Crop سه خوانش داریم: Native برابر ۱۸٪، OCR برابر ۱۸٪ و VLM برابر ۸۱٪. این مثال فرضی است. توافق دو موتور یک سیگنال مفید است اما اثبات درستی نیست؛ ممکن است منبع مشترک یا الگوی خطای مشترک داشته باشند. از Consensus برای کشف تعارض استفاده کنید و با شواهد معتبر همان ناحیه تصمیم بگیرید. در نمونهٔ حساس، وضعیت نامعلوم بهتر از پذیرش یک حدس است.

## ۱۱. تشخیص خطا و مسیر بازیابی؛ به‌جای تعویض تصادفی مدل‌ها

برای حل یک خطا باید بتوانیم مرز مرحله‌ها را ببینیم. اگر خروجی Markdown اشتباه است، معلوم نیست علت OCR، Layout، Reading Order، IR یا خود تبدیل به Markdown بوده باشد. به همین دلیل نگهداری Artifact میانی (مثلاً تصویر صفحه، کادرها، متن خام، خروجی مدل و ساختار بازسازی‌شده) ارزش زیادی دارد.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-19-title" direction="ltr" height="508" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 508" width="720" xml:lang="fa"><title id="v5-figure-19-title">مسیر تشخیص نخستین مرحلهٔ خطادار</title><rect x="0" y="0" width="720" height="508" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="23" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="25">اولین مرحلهٔ خراب را پیدا کنید</text><rect fill="#e1f1ef" height="58" rx="10" stroke="#176e73" width="300" x="16" y="48"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="166" y="82">ناحیه جا افتاده؟</text><path d="M316 77H370m-6 -5 6 5 -6 5" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="342" y="65">بله</text><rect fill="#fffdf9" height="58" rx="10" stroke="#aab8b5" width="334" x="370" y="48"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="82">چیدمان، مسیریابی، پوشش</text><path d="M166 106V133m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="188" y="124">خیر</text><rect fill="#e1f1ef" height="58" rx="10" stroke="#176e73" width="300" x="16" y="134"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="166" y="168">متن خام اشتباه است؟</text><path d="M316 163H370m-6 -5 6 5 -6 5" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="342" y="151">بله</text><rect fill="#fffdf9" height="58" rx="10" stroke="#aab8b5" width="334" x="370" y="134"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="168">خواندن PDF، تصویر یا OCR</text><path d="M166 192V219m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="188" y="210">خیر</text><rect fill="#e1f1ef" height="58" rx="10" stroke="#176e73" width="300" x="16" y="220"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="166" y="254">ترتیب یا سلول اشتباه است؟</text><path d="M316 249H370m-6 -5 6 5 -6 5" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="342" y="237">بله</text><rect fill="#fffdf9" height="58" rx="10" stroke="#aab8b5" width="334" x="370" y="220"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="254">ترتیب خواندن / ساختار جدول</text><path d="M166 278V305m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="188" y="296">خیر</text><rect fill="#e1f1ef" height="58" rx="10" stroke="#176e73" width="300" x="16" y="306"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="166" y="340">IR درست ولی خروجی خراب است؟</text><path d="M316 335H370m-6 -5 6 5 -6 5" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="342" y="323">بله</text><rect fill="#fffdf9" height="58" rx="10" stroke="#aab8b5" width="334" x="370" y="306"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="340">تبدیل خروجی، Chunking یا Index</text><path d="M166 364V391m-5 -6 5 6 5 -6" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="188" y="382">خیر</text><rect fill="#e1f1ef" height="58" rx="10" stroke="#176e73" width="300" x="16" y="392"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="166" y="426">شواهد کافی نیست؟</text><path d="M316 421H370m-6 -5 6 5 -6 5" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="rtl" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="342" y="409">بله</text><rect fill="#fffdf9" height="58" rx="10" stroke="#aab8b5" width="334" x="370" y="392"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="16" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="537" y="426">بازبینی / قرنطینه</text></g></svg><figcaption>نقشهٔ تشخیص ریشهٔ خطا: مرحله‌ها را از روی شواهد بررسی کنید. اگر اولین نشانه در یک لایه پیدا شد، همان لایه را جدا آزمایش کنید؛ ممکن است خطاهای دیگری هم وجود داشته باشند.</figcaption></figure>

برای هر خطا اول بررسی کنید آیا عنصر اصلی وجود دارد، آیا ناحیه پیدا شده، آیا متن خام درست است، آیا ساختار و ارتباط‌ها درست‌اند و آیا همان داده به خروجی نهایی رسیده است. اولین لایه‌ای که در مقایسه با منبع شکست می‌خورد، نقطهٔ شروع بررسی است؛ ممکن است بیش از یک مشکل هم وجود داشته باشد.

چند سیگنال مفید برای پیدا کردن ناحیهٔ مشکوک عبارت‌اند از نویسهٔ نامعتبر، هم‌پوشانی متن و تصویر با کادرهای Layout، سلول ظاهراً خالی دارای محتوای تصویری، صفحه یا ناحیهٔ جاافتاده، فرمول تشخیص‌داده‌نشده و تعارض عددی. برای مثال، اگر Parser یک سلول را خالی تشخیص داد، محدودهٔ همان سلول را روی تصویر بررسی کنید. البته اگر مقدار واقعی ۱۸ باشد و OCR عدد ۱۹ بدهد، هر دو عدد از نظر Unicode و Schema معتبرند؛ قواعد ساختاری ممکن است هیچ خطایی پیدا نکنند. برای چنین خطاهایی به شواهد تصویری یا مرجع مستقل نیاز داریم.

### نقشهٔ عیب‌یابی عملی: ۳۷ خانوادهٔ مسئله

این ۳۷ خانوادهٔ خطا از تجربهٔ توسعه و آزمایش اسناد فارسی جمع‌بندی شده‌اند؛ یک Taxonomy جهانی یا فهرست کامل همهٔ خطاها نیستند. از ستون تشخیص برای پیدا کردن اولین مرحلهٔ خراب و از ستون راهکار و بررسی برای طراحی تست استفاده کنید.

#### الف) ورودی، تشخیص و خواندن متن

| مشکل / نشانه | فرضیهٔ ریشه‌ای و روش تشخیص | مسیر اصلاح و آزمون پذیرش |
|---|---|---|
| فایل با پسوند معتبر باز نمی‌شود | پسوند با فرمت واقعی یا محتوای فایل نمی‌خواند؛ Header و قابلیت بازشدن را بررسی کنید | شناسایی فرمت واقعی، محدودیت منابع و رد فایل نامعتبر؛ تست فایل ناقص و با پسوند جعلی |
| PUA یا نویسهٔ خراب | نگاشت فونت یا Decode؛ متن خام را با تصویر و Decoder دوم مقایسه کنید | نگاشت معتبر یا OCR ناحیه؛ تطابق نویسه‌ها با مرجع |
| متن خراب از کنترل کیفیت عبور می‌کند | کنترل فقط غیرخالی بودن متن را می‌سنجد | بررسی ناحیه‌ای PUA/نویسهٔ جایگزین و شواهد معنایی؛ تست نمونهٔ خرابِ غیرخالی |
| رونویسی تصویری فارسی ضعیف است | زبان مدل، فونت، دامنه یا Recognition؛ Crop و خروجی خام را ببینید | مقایسهٔ هم‌شرایط موتورهای OCR؛ CER/WER و تطابق علامت‌های حساس |
| بعضی ناحیه‌ها اصلاً پردازش نمی‌شوند | Routing یا Detection ناقص؛ تعداد و شناسهٔ ناحیه‌های مورد انتظار را مقایسه کنید | ثبت فهرست ناحیه‌ها و وضعیت تک‌تک؛ تست پوشش کامل |
| متن ریز در Crop جا می‌افتد | رزولوشن، حاشیه یا Resize نامناسب | مقایسهٔ Crop و تصویر مرجع با DPI متفاوت؛ کنترل حافظه و صحت متن ریز |
| ترتیب RTL یا ستون‌ها خراب است | ترتیب منطقی رشته یا Reading Order صفحه؛ هر سطح را جدا بررسی کنید | جداسازی ترتیب متن و صفحه؛ تست فارسی چندستونه و عبارت ترکیبی |
| عدد یا علامت معکوس شده | خطا در OCR، Decode یا اصلاح Bidi؛ خام و نهایی را مقایسه کنید | اصلاح بر اساس شواهد همان ناحیه؛ تست اعداد درست و نادرست، بدون Reverse سراسری |
| درصد، واحد یا بازه حذف شده | نویسهٔ ریز، نرمال‌سازی یا Chunking | حفظ Raw و واحد؛ Exact Match برای رقم، علامت، واحد و رابطهٔ آن |
| فرمول تبدیل به متن بی‌معنی شده | تشخیص نوع ناحیه یا بازسازی نمادها خراب است | تشخیص فرمول، خوانش تخصصی یا بازبینی؛ مقایسهٔ رابطهٔ نمادها با مرجع |
| تصویر، نمودار یا Caption نادیده گرفته شده | Detection یا خروجی فیلترشده؛ Overlay و فهرست عناصر را بررسی کنید | حفظ ناحیه، Caption و دادهٔ قابل‌مشاهده؛ تست Layout Recall و اتصال Caption |
| ساختار DOCX ناقص است | تبدیل ساده به متن یا PDF، ساختار Word را از دست داده است | خواندن Style، فهرست، جدول و روابط با ابزار Native؛ تست اسناد Word مرجع |

#### ب) هندسه و ساختار سند

| مشکل / نشانه | فرضیهٔ ریشه‌ای و روش تشخیص | مسیر اصلاح و آزمون پذیرش |
|---|---|---|
| تعداد سلول یا هندسه غلط | Detector/TSR یا تطبیق متن با Grid؛ Overlay بسازید | اصلاح ساختار؛ تست تعداد سطر، ستون و مرز سلول |
| مقدار زیر سرستون غلط | ارتباط Header از دست رفته | نگهداری Header ID و مسیر سرستون چندسطحی؛ تست ارتباط دقیق |
| دو جدول مجاور ادغام شده‌اند | فاصلهٔ هندسی تنها معیار بوده | مرز، عنوان و شناسهٔ مستقل؛ تست عدم ادغام جدول مستقل |
| دو جدول با معنی متفاوت مخلوط شده‌اند | ادغام بدون توجه به بخش سند | تطبیق Caption، موضوع و ساختار؛ تست عضویت هر ردیف در جدول صحیح |
| جدول ادامه‌دار تکه‌تکه شده | ارتباط صفحات تشخیص داده نشده | کاندیدای اتصال با سرستون و هندسه؛ تست ادامهٔ واقعی و نمونهٔ منفی |
| سرستون تکراری به‌عنوان داده یا سطر شکسته حذف شده | تشخیص Header/Continuation ناقص | ثبت سطر مرزی و سرستون؛ تست شمارش ردیف و تداوم مقدار |
| عنوان جدول به جدول کناری وصل شده | ارتباط Caption و Layout غلط | رابطهٔ صریح عنوان با جدول؛ تست چند جدول نزدیک |
| پاورقی یا واحد به ستون غلط وصل شده | رابطهٔ معنایی در IR ذخیره نشده | پیوند پاورقی و واحد به محدودهٔ معتبر؛ تست ارجاع درست |
| فرم خالی، شکست OCR فرض شده | نبود متن با نبود ساختار اشتباه شده | حفظ فیلدها و حالت خالی/ناخوانا؛ تست فرم خالی و پرشده |
| فهرست یا سلسله‌مراتب بندها حذف شده | خطا در Heading/List Reconstruction | بازسازی درخت و شماره‌گذاری؛ تست بند و زیر‌بند و شمارهٔ ترکیبی |
| شمارهٔ صفحه در استناد غلط است | شمارهٔ فیزیکی و چاپی یکسان فرض شده | ذخیرهٔ هر دو شماره؛ باز کردن صفحهٔ استنادشده و تطبیق با اصل |

#### ج) نمایش میانی، کیفیت و انتشار

| مشکل / نشانه | فرضیهٔ ریشه‌ای و روش تشخیص | مسیر اصلاح و آزمون پذیرش |
|---|---|---|
| TableIR اصلاح شده، خروجی مشتق نشده | مسیر Export از نسخهٔ قدیمی می‌خواند | تولید همهٔ خروجی‌ها از نسخهٔ مشخص؛ تست تغییر سلول و بازسازی نماها |
| سلول ادغام‌شده هنگام Chunking شکسته | قطعه‌بندی بدون رابطهٔ Span و Header | Chunk ساختارمحور و تکرار زمینهٔ لازم؛ تست retrieval برای ردیف‌های مرتبط |
| Markdown ساختار جدول را خراب می‌کند | فرمت ساده ظرفیت حفظ ساختار کامل ندارد | IR/HTML را نگه دارید؛ تست معادل بودن معنی و ارتباط در خروجی لازم |
| شناسه‌ها ناپایدار و Provenance ناقص | اتصال به ترتیب موقت یا مختصات تبدیل‌شده | شناسهٔ نسخه‌دار و نگاشت مختصات؛ تست ارجاع پس از پردازش مجدد |
| دادهٔ تشخیصی خراب قابل جست‌وجو شده | مسیر Debug و انتشار مشترک است | جداسازی Candidate و Accepted؛ تست عدم ورود دادهٔ ردشده به ایندکس |
| ناحیهٔ قرنطینه‌شده پنهان مانده | فیلتر خروجی، وضعیت را حذف می‌کند | گزارش Coverage و دلیل عدم انتشار؛ تست شمارش همهٔ ناحیه‌ها |
| مرجع مستقل وجود ندارد | خروجی Parser به جای Gold استفاده شده | برچسب‌گذاری از روی منبع و بازبینی؛ تست مرجع مستقل و ثابت |
| Parser و ایندکس ناسازگارند | انتشار چندمرحله‌ای بدون نسخهٔ هماهنگ | Index version و انتشار اتمیک یا تغییر Alias؛ تست جست‌وجوی نسخهٔ جدید |

#### د) زیرساخت و اجرای Pipeline

| مشکل / نشانه | فرضیهٔ ریشه‌ای و روش تشخیص | مسیر اصلاح و آزمون پذیرش |
|---|---|---|
| پاسخ مدل بعد از خطای Validation گم شده | خروجی خام قبل از اعتبارسنجی ذخیره نشده | ذخیرهٔ پایدار قبل از Validation؛ تست Crash و Replay |
| Cache خروجی قدیمی می‌دهد | کلید Cache ناقص یا Replay نامشخص است | هش محتوا، نسخهٔ موتور و تنظیمات؛ تست تغییر هر ورودی مؤثر |
| Job تکراری یا Artifact گم شده | اجرای هم‌زمان، Race یا Retry بدون Idempotency | کلید عملیات، قفل یا کنترل هم‌زمانی و ثبت نتیجه؛ تست Retry و Crash |
| هویت مدل، Endpoint یا خطا مبهم است | Observability ناقص | ثبت نسخه، مدل واقعی، مسیر اجرا و خطای مرحله؛ تست پاسخ خطا و Timeout |
| Force به معنی استخراج تازه نیست | اجرای مجدد فقط Export یا Replay می‌کند | عملیات صریح Fresh/Rebuild/Verify؛ تست عدم استفاده از Cache در Fresh |
| پردازش ناقص موفق گزارش شده | موفقیت کلی با پایان Job یکی فرض شده | وضعیت هر مرحله و Coverage؛ تست شکست یک صفحه و عدم پذیرش کل سند |

این جدول یک نقشهٔ تشخیصی است. وقتی خطای جدیدی پیدا می‌کنید، از همین ساختار استفاده کنید: نشانه، شواهد، فرضیه، تغییر محدود و معیار تأیید. به این ترتیب هر شکست می‌تواند به یک Regression Test تبدیل شود.

## ۱۲. Provenance، اعتبارسنجی و قابلیت اعتماد؛ چه چیزی اجازهٔ انتشار دارد؟

### منشأ اطلاعات را تا منبع حفظ کنید

Provenance یعنی اطلاعاتی دربارهٔ منشأ و مسیر تولید یک داده. اگر مدل زبانی در جوابش می‌گوید «سقف مبلغ دو میلیارد ریال است»، باید بتوانیم از آن جمله به Chunk، سلول یا پاراگراف، ناحیهٔ صفحه و فایل اصلی برگردیم. این کار هم برای اعتماد و هم برای دیباگ لازم است.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-20-title" direction="ltr" height="518" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 518" width="720" xml:lang="fa"><title id="v5-figure-20-title">زنجیره منشأ اطلاعات از فایل اصلی تا پاسخ هوش مصنوعی.</title><defs><marker id="arrow-20" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="518" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#e1f1ef" height="48" rx="8" stroke="#176e73" width="440" x="140" y="10"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="40">سند منبع و هش</text><path d="M360 58L360 77" fill="none" marker-end="url(#arrow-20)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="48" rx="8" stroke="#aab8b5" width="440" x="140" y="78"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="108">صفحهٔ فیزیکی / چاپی</text><path d="M360 126L360 145" fill="none" marker-end="url(#arrow-20)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="48" rx="8" stroke="#aab8b5" width="440" x="140" y="146"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="176">ناحیه و کادر هندسی</text><path d="M360 194L360 213" fill="none" marker-end="url(#arrow-20)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="48" rx="8" stroke="#aab8b5" width="440" x="140" y="214"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="244">جدول، سلول، سطر و ستون</text><path d="M360 262L360 281" fill="none" marker-end="url(#arrow-20)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="48" rx="8" stroke="#aab8b5" width="440" x="140" y="282"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="312">مقدار استخراج‌شده و وضعیت</text><path d="M360 330L360 349" fill="none" marker-end="url(#arrow-20)" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="48" rx="8" stroke="#aab8b5" width="440" x="140" y="350"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="380">قطعه و شاهد</text><path d="M360 398L360 417" fill="none" marker-end="url(#arrow-20)" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="48" rx="8" stroke="#176e73" width="440" x="140" y="418"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="448">پاسخ AI و استناد</text></g></svg><figcaption>زنجیره منشأ اطلاعات از فایل اصلی تا پاسخ هوش مصنوعی.</figcaption></figure>

در یکی از نمونه‌های قبلی، صفحهٔ فیزیکی هشتم PDF با صفحهٔ هفتم چاپ‌شده در سند مطابقت داشت. اگر شماره‌ها را یکی فرض کنیم، استناد حتی با متن درست هم می‌تواند به صفحهٔ دیگری برود.

برای هر عنصر مهم، حداقل هویت سند، شمارهٔ صفحه، ناحیه، نسخهٔ Pipeline، روش خواندن و وضعیت اعتبار را نگه دارید. برای اصلاح انسانی هم ثبت کنید چه چیزی، چرا و بر اساس کدام شواهد تغییر کرده است. اطلاعات محرمانهٔ سند را بی‌دلیل در لاگ عمومی کپی نکنید.

### Validation را به JSON Schema محدود نکنید

| لایهٔ اعتبارسنجی | چه چیزی را می‌سنجد؟ |
|---|---|
| Schema | نوع داده، فیلدهای لازم، شناسه و قالب پاسخ |
| Structural | پوشش صفحات، هندسه، ترتیب، شبکهٔ جدول، Span و ارتباط عناصر |
| Content | تطابق متن و مقدار با شواهد؛ بررسی علامت و عدد |
| Domain | قواعد معتبر کاربرد، مثل واحد و بازه؛ نه ساخت مقدار از روی حدس |
| Publication | وضعیت همهٔ بخش‌ها، نسخه و صلاحیت ورود به RAG |

یک Numeric Cross-check یا کنترل جمع جدول می‌تواند تناقض را پیدا کند، اما الزاماً مقدار درست را تعیین نمی‌کند. اگر جمع سازگار است، هنوز ممکن است چند عدد به‌طور هم‌زمان اشتباه باشند. قواعد دامنه‌ای برای کشف مسئله‌اند؛ جایگزین شواهد منبع نیستند.

### Quarantine؛ چطور بگوییم این قسمت را نمی‌دانیم؟

اگر ناحیه‌ای قابل اعتماد نیست، نباید صرفاً برای تمیز به نظر رسیدن سند آن را حذف کنیم. باید وضعیتش را ثبت کنیم و مشخص باشد آیا دادهٔ معتبر قابل انتشار از همان سند داریم یا نه.

| وضعیت | معنی |
|---|---|
| Candidate | خروجی اولیه؛ هنوز معتبر اعلام نشده |
| Review Required | شواهد ناکافی یا تعارض؛ نیاز به بررسی |
| Accepted | معیارهای تعریف‌شده برای آن کاربرد و نسخه را گذرانده |
| Quarantined | محتوای نامعتبر یا ناقص؛ در مسیر دانش معتبر منتشر نمی‌شود |
| Failed | اجرا یا پردازش ناموفق بوده و دلیل آن ثبت شده است |

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-21-title" direction="ltr" height="358" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 358" width="720" xml:lang="fa"><title id="v5-figure-21-title">حسابداری پوشش ناحیه‌ها</title><rect x="0" y="0" width="720" height="358" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="330" rx="8" width="720" x="0" y="0"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="360" y="32">حسابداری پوشش ناحیه‌ها</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="52"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="82">ناحیهٔ مورد انتظار → خوانش موجود یا مفقود</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="116"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="146">خوانش موجود → بررسی شواهد و تعارض</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="180"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="210">پذیرفته‌شده / نیازمند بازبینی / قرنطینه</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="244"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="274">ثبت وضعیت همهٔ ناحیه‌ها، حتی شکست</text></g></svg><figcaption>[مسیر پیشنهادی] پاسخ موفق API را با پذیرش یکی ندانید.</figcaption></figure>

تعداد پاسخ‌های موفق API با تعداد نواحی معتبر یکی نیست. Missing، Conflict و Failure باید قابل مشاهده باشند. اگر یک ناحیهٔ حل‌نشده ممکن است معنای بند یا جدول دیگری را عوض کند، انتشار بخش‌های وابسته را هم متوقف کنید. در کاربرد کم‌ریسک انتشار جزئی با برچسب روشن ممکن است کافی باشد؛ در کاربرد حساس گاهی باید کل سند متوقف شود.

## ۱۳. ارزیابی؛ به‌جای رتبه‌بندی مدل‌ها، اجزای Pipeline را آزمایش کنیم

وقتی می‌خواهیم بین دو Parser تصمیم بگیریم، اولین سؤال معمولاً این است که کدام مدل بهتر است. اما ممکن است یکی از Pipelineها Layout خوبی داشته باشد و OCR نامناسب، و دیگری OCR خوب و بازسازی جدول ضعیف. اگر همهٔ اجزا را با هم عوض کنیم، نمی‌فهمیم چه چیزی واقعاً باعث تغییر نتیجه شده است.

### Gold Dataset و طراحی مجموعهٔ ارزیابی

Gold Dataset مجموعه‌ای از نمونه‌های ورودی با خروجی مرجع مستقل و بررسی‌شده است. اگر خروجی یک Parser قبلی را بدون بررسی منبع به‌عنوان مرجع بگیریم، ممکن است خطاهای قبلی وارد Benchmark شوند. همین مسئله برای داوری صرفاً با مدل زبانی هم وجود دارد.

برای اسناد فارسی، دیتاست را به خانواده‌های واقعی تقسیم کنید. متن ساده، جدول پیچیده، فرم خالی، صفحهٔ اسکن‌شده، چندستونه، فونت قدیمی، تصویر مهرخورده، فرمول، تاریخ شمسی، ارقام ترکیبی، پاورقی، سربرگ و جدول چندصفحه‌ای هر کدام باید نماینده داشته باشند. هم نمونهٔ ساده و هم نمونهٔ دشوار را نگه دارید. نمونه‌های بسیار مشابه از یک قالب را بین مجموعهٔ توسعه و تست پخش نکنید؛ تقسیم را تا حد امکان در سطح سند یا خانوادهٔ قالب انجام دهید.

برای هر نمونه مشخص کنید چه عناصری باید وجود داشته باشند، چه متن و روابطی صحیح‌اند و چه قسمت‌هایی واقعاً خواندنی نیستند. اختلاف برچسب‌زن‌ها را حل و نسخهٔ مرجع را ثابت کنید. دادهٔ مصنوعی با جواب معلوم برای تولید حالت‌های مرزی و Regression مفید است، اما نمی‌تواند به‌تنهایی رفتار عجیب فونت‌ها، اسکن‌ها و ساختارهای واقعی را پوشش دهد.

### یک پروتکل کوچک و قابل اشتراک برای Gold فارسی مالی/سازمانی

**[مسیر پیشنهادی]** با ۵۰ تا ۱۰۰ صفحهٔ واقعی مجاز شروع کنید؛ برای نمونه ۸۰ صفحه در شش گروه زیر. این تخصیص فقط نقطهٔ شروع است، نه حداقل آماری اثبات‌شده یا دیتاست ساخته‌شده. گروه اصلی هر صفحه را برای شمارش تعیین کنید و ویژگی‌های هم‌پوشان مثل RTL، فونت قدیمی، مهر، چرخش یا اسکن کم‌کیفیت را جدا برچسب بزنید. جدول چندصفحه‌ای را به‌صورت یک واحد مرتبط نگه دارید، نه دو نمونهٔ مستقل در Train و Test.

| گروه اصلی نمونه | تعداد پیشنهادی صفحه | شاهدی که باید ثبت شود |
|---|---:|---|
| PDF دیجیتال سالم و ساده | ۱۵ | متن منطقی، عنوان و مختصات |
| PDF ترکیبی یا لایهٔ OCR قدیمی | ۱۵ | تفکیک ناحیهٔ Native و تصویری، ناحیهٔ جاافتاده |
| اسکن متن چاپی | ۱۵ | متن خام مرجع، علائم و ترتیب خطوط |
| دست‌خط، مهر و عکس دشوار | ۱۰ | فیلدهای قابل خواندن، ناحیهٔ واقعاً نامعلوم |
| جدول پیچیدهٔ تک‌صفحه | ۱۵ | هندسه، Span، Header و مقدار/واحد |
| جدول ادامه‌دار و چندصفحه‌ای | ۱۰ | ارتباط صفحات، سرستون و ردیف شکسته |

برای هر صفحه، دو بازبین مستقل متن منطقی و عناصر را از روی تصویر اصلی ثبت کنند؛ اختلاف را با شاهد حل کنید. برای ناحیهٔ ناخوانا وضعیت `unreadable` را ثبت کنید، نه حدس. واحد پول، علامت، تاریخ، شناسه، رقم، سلول و ارتباط سرستون را به‌صورت جدا علامت بزنید. همراه داده نسخهٔ دستورالعمل، شناسهٔ سند و صفحه، مختصات، مرز Crop، هش ورودی، شیوهٔ نرمال‌سازی، نسخهٔ مرجع و مجوز انتشار را نگه دارید. تمام صفحات یک سند/قالب را در یک Split قرار دهید؛ نمونه‌های استفاده‌شده برای اصلاح Router یا پرامپت را وارد Test مستقل نکنید. اگر دامنهٔ کاربرد گسترش یافت، تعداد و تنوع را افزایش دهید.

برای اشتراک، فقط سند عمومی یا نمونه‌ای با اجازهٔ صریح منتشر کنید؛ حذف نام از تصویر به تنهایی ممکن است اطلاعات مالی، شناسهٔ پنهان یا متادیتا را پاک نکند. برای سند خصوصی می‌توانید دستورالعمل برچسب‌زنی، Schema، معیارها، نمونهٔ مصنوعی و نتایج تجمیعی مجاز را منتشر کنید، بدون انتشار اصل فایل. **Persian Pixel** بیش از ۳۴۳ هزار جفت مصنوعی تصویر و متن در سطح جمله، پاراگراف و صفحه گزارش می‌کند؛ برای Stress Test فونت و تخریب تصویر و آماده‌سازی آموزش مفید است. مجوز و شرایط دسترسی Dataset Card را بررسی کنید؛ دادهٔ مصنوعی را از Gold واقعی جدا نگه دارید. [۳۸](#ref-38)

### معیارها را براساس مؤلفه و کاربرد انتخاب کنید

| معیار | چه چیزی را نشان می‌دهد؟ |
|---|---|
| CER / WER | فاصلهٔ ویرایشی نویسه‌ها یا کلمات نسبت به طول مرجع؛ هر چه کمتر بهتر؛ سیاست نرمال‌سازی را ثبت کنید |
| Exact Match / Critical Token Accuracy | تطابق کامل رشته یا رقم، علامت، شناسه، تاریخ و واحد حساس |
| Layout Precision / Recall | چند ناحیهٔ تشخیص‌داده‌شده درست‌اند و چند ناحیهٔ مرجع پیدا شده‌اند؛ با قاعدهٔ تطبیق هندسی ثابت |
| Reading Order | تطابق توالی یا روابط تقدم عناصر با ترتیب مرجع |
| BiDi / Logical-order accuracy | سهم Runها یا خطوط دارای ترتیب منطقی صحیح از کل Runها یا خطوط مرجعِ برچسب‌خورده؛ مورد جاافتاده ناموفق؛ جدا از Reading Order بلوک و ظاهر UI |
| Mixed-script token exact match | سهم توکن‌های حساس ترکیبی (مثل شناسهٔ لاتین کنار فارسی، تاریخ، علامت و عدد) با تطابق کامل مقدار، ترتیب داخلی و ارتباط با فیلد از کل توکن‌های مرجع؛ توکن جاافتاده مردود |
| Table structure / TEDS | شباهت ساختار جدول؛ TEDS مبتنی بر فاصلهٔ ویرایشی درخت جدول است؛ تعریف نسخهٔ معیار را مشخص کنید |
| Cell content / Header Association | درستی متن سلول و تعلق آن به سطر، سرستون و واحد صحیح |
| Coverage / Review Rate / Quarantine Rate | پوشش = عناصر مرجع دارای Candidate / کل عناصر مرجع؛ نرخ بازبینی و قرنطینه = تعداد رکوردهای یکتای نیازمند بازبینی یا قرنطینه / کل رکوردهای هدف؛ وضعیت نهایی را در لحظهٔ تعریف‌شده بسنجید |
| False Acceptance Rate | سهم خروجی‌های غلط که سیستم به اشتباه قبول کرده؛ مخرج (مثلاً از کل خروجی‌های پذیرفته‌شده یا کل موارد غلط) را صریح تعریف کنید |
| Latency / Throughput / Cost | زمان، تعداد صفحه در واحد زمان، حافظه، هزینهٔ هر صفحه و هزینهٔ هر رکورد پذیرفته‌شده پس از Retry و بازبینی؛ Cache-hit را از اجرای تازه جدا کنید؛ اگر هیچ رکوردی پذیرفته نشد، هزینهٔ هر پذیرش تعریف‌نشده است |

OmniDocBench نمونه‌ای از ارزیابی چندبعدی متن، جدول، فرمول و ترتیب خواندن است؛ OHR-Bench هم اثر نویز OCR روی RAG را جداگانه مطالعه می‌کند. [۱۰](#ref-10) [۱۱](#ref-11) این‌ها منابع خوبی برای یاد گرفتن طراحی معیارند، ولی امتیاز یک مدل در Benchmark عمومی تضمین نمی‌کند روی اسناد فارسی سازمان شما خوب عمل کند.

برای آزمون BiDi، پیش از ارزیابی قرارداد هر Run را بنویسید: متن مرجع در ترتیب منطقی Unicode، جهت پایه، دنبالهٔ مورد انتظار عدد و نشانه‌ها، و در صورت نیاز موقعیت فیزیکی. پس از تطبیق Region، خطای recognition را با CER خام گزارش کنید و در معیار ترتیب، توکن‌های همسان‌شده را جدا بسنجید تا نویسهٔ غلط لزوماً خطای Bidi شمرده نشود. برای جلوگیری از نمرهٔ خوش‌بینانه، نرخ Runهای نامنطبق/ناپیدا را هم کنار نمره گزارش کنید؛ در تطابق کامل ترکیبی، هر مورد جاافتاده شکست است. یک نمرهٔ ترتیب پیشنهادی، نسبت زوج‌های تقدم درست از زوج‌های مرجع قابل تطبیق است؛ تعداد زوج‌های قابل تطبیق و Coverage را نیز اعلام کنید. تست تصویری UI جداگانه با نمونهٔ مرجع انجام شود. این تعریف‌ها پیشنهاد اجرایی‌اند، نه استاندارد واحد پذیرفته‌شده برای BiDi فارسی.

### چرا عدد ۹۹٫۸٪ می‌تواند گمراه‌کننده باشد؟

فرض کنید در یک سند آموزشیِ فرضی، از ۱۰٬۰۰۰ واحد متنی، ۹٬۹۸۰ واحد درست باشند. دقت کلی بر اساس تطابق همان واحدها ۹۹٫۸٪ است. حالا اگر در همان سند ۳۰ واحد حساس داشته باشیم و پنج مورد از آن‌ها غلط باشند، دقت بخش حساس ۲۵ از ۳۰ یعنی ۸۳٫۳۳٪ است. این یک مثال فرضی است، نه نتیجهٔ Benchmark.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-22-title" direction="ltr" height="270" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 270" width="720" xml:lang="fa"><title id="v5-figure-22-title">مقایسهٔ فرضی دقت کلی و دقت فیلدهای حساس</title><rect x="0" y="0" width="720" height="270" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="22" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="30">دقت کلی در برابر دقت فیلد حساس</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="120" y="80">همهٔ واحدها</text><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="120" y="104">9,980 / 10,000</text><rect fill="#aab8b5" height="44" rx="5" stroke="none" width="442" x="232" y="64"/><rect fill="#176e73" height="44" rx="5" stroke="none" width="441.116" x="232" y="64"/><text direction="ltr" fill="#ffffff" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="631.116" y="92">99.80%</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="600" text-anchor="middle" unicode-bidi="isolate" x="120" y="170">واحدهای حساس</text><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="120" y="194">25 / 30</text><rect fill="#aab8b5" height="44" rx="5" stroke="none" width="442" x="232" y="154"/><rect fill="#94600d" height="44" rx="5" stroke="none" width="368.33333186" x="232" y="154"/><text direction="ltr" fill="#ffffff" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="558.33333186" y="182">83.33%</text><path d="M232.0 204v5" fill="none" stroke="#5f6964" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="232.0" y="225">0%</text><path d="M342.5 204v5" fill="none" stroke="#5f6964" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="342.5" y="225">25%</text><path d="M453.0 204v5" fill="none" stroke="#5f6964" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="453.0" y="225">50%</text><path d="M563.5 204v5" fill="none" stroke="#5f6964" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="563.5" y="225">75%</text><path d="M674.0 204v5" fill="none" stroke="#5f6964" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><text direction="ltr" fill="#5f6964" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="12" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="674.0" y="225">100%</text></g></svg><figcaption>مثال فرضی: دقت کلی ۹۹٫۸٪ و دقت واحدهای حساس ۸۳٫۳۳٪. این اعداد نتیجهٔ آزمایش واقعی نیستند.</figcaption></figure>

برای یک کتاب آموزشی شاید بعضی خطاهای جزئی در فاصله‌ها چندان مهم نباشند. برای سندی که مبنای یک محاسبه یا تصمیم حقوقی است، خطا در علامت، مقیاس یا رابطهٔ سلول ممکن است بسیار مهم باشد. معیار کلی را همراه با معیار خانواده‌های خطا گزارش کنید و برای کاربرد حساس، شرط پذیرش مستقل داشته باشید.

### Ablation Study؛ ارزش هر مرحله را اندازه بگیرید

Ablation یعنی یک جزء را حذف یا عوض کنیم تا اثر آن را روی کیفیت و هزینهٔ سیستم ببینیم. در شروع، یک Baseline تعریف کنید و روی همان مجموعهٔ ثابت ارزیابی کنید. مثلاً:

| آزمایش پیشنهادی | تغییر نسبت به مرجع |
|---|---|
| A | Native + ساختار پایه؛ بدون بازیابی اضافه |
| B | A + OCR روی نواحی نیازمند آن |
| C | A + VLM روی نواحی تعریف‌شده |
| D | A + OCR + VLM؛ بررسی اثر متقابل دو مرحله |
| E | جایگزینی Layout Model در B با ثابت ماندن سایر اجزا |
| F | مقایسهٔ دو روش Table Recognition با متن مرجع یکسان |
| G | مقایسهٔ چند DPI و پیش‌پردازش با Recognizer ثابت |
| H | مقایسهٔ روش تطبیق و کنترل کیفیت با کاندیداهای خام ثابت |

این‌ها طرح آزمایش‌اند، نه ترکیب‌های تأییدشده. برای مقایسهٔ D با B یا C، یک تغییر مشخص را جدا کنید؛ اگر چند چیز هم‌زمان تغییر کند، نسبت دادن بهبود به یک جزء دشوار است. برای Router ابتدا مسیرهای ممکن را روی نمونه‌ها مقایسه کنید، سپس خود Router را با همان نمونه‌های توسعه ارزیابی کنید و نتیجهٔ نهایی را روی مجموعهٔ مستقل بسنجید.

فقط میانگین را گزارش نکنید. خروجی‌های ناموفق و جاافتاده را هم در مخرج پوشش و کیفیت لحاظ کنید. نسخهٔ مدل، Prompt، DPI، تنظیمات، زمان، دستگاه، Cache و دیتاست را ثبت کنید. برای مقایسهٔ APIها هزینهٔ واقعی، محدودیت نرخ و خطای اجرا را هم در نظر بگیرید.

### یک آزمایش تصمیم‌ساز به جای مقایسهٔ تبلیغاتی مدل‌ها

برای اینکه واقعاً بفهمیم VLM مستقیم به‌اندازهٔ Parser کافی است یا نه، قبل از آزمایش یک خروجی یکسان تعریف کنیم. اگر محصول فاکتور است، همهٔ مسیرها باید همان فیلدها، ردیف‌ها و شواهد را تحویل دهند. اگر محصول RAG است، علاوه بر کیفیت استخراج، با Retriever و مدل پاسخ‌دهندهٔ ثابت بسنجیم آیا شاهد درست و پاسخ صحیح پیدا می‌شود. اینکه یک مسیر فقط چند فیلد بدهد و مسیر دیگر کل سند را، مقایسهٔ هزینه یا سرعت خام آن‌ها را گمراه‌کننده می‌کند.

| مسیر مقایسه | چه فرضیه‌ای را بررسی می‌کنیم؟ |
|---|---|
| Native یا خوانندهٔ ساختار فایل | چقدر از نیاز محصول بدون هیچ مدل تصویری حل می‌شود؟ |
| Native + Layout / Table | خطای اصلی ساختار است یا خوانش متن؟ |
| OCR و Layout تخصصی | آیا مسیر ارزان و سبک برای نواحی تصویری کافی است؟ |
| VLM تخصصی صفحه به ساختار | آیا یک مدل تخصصی می‌تواند چند مسئولیت را یک‌جا انجام دهد؟ |
| VLM عمومی مستقیم (مدل سریع و مدل قوی‌تر به‌طور جداگانه) | آیا افزایش توان فهم بصری، کیفیت فیلد یا پوشش را بهتر می‌کند؟ |
| مسیر ترکیبی Native + VLM-first برای نواحی تصویری فارسی | آیا حفظ متن سالم و خوانش هدفمند همان کیفیت را با هزینهٔ کمتر ایجاد می‌کند؟ |
| خروجی صحیح از انسان یا Gold | سقف کیفیت downstream با ورودی سالم چقدر است و مشکل واقعاً از Parser است؟ |

همان فایل‌ها، هدف، نسخه، سیاست نرمال‌سازی، سقف خروجی و قواعد پذیرش را ثابت نگه دارید. برای مواردی که مدل عمومی نتیجه‌ای متفاوت می‌دهد، چند بار اجرا را هم بررسی کنید؛ برای مسیرهای محلی سخت‌افزار و Cold/Warm بودن را ثبت کنید. اگر مدل اول در یک نوع سند از نظر صحت ضعیف‌تر ولی خیلی سریع‌تر است، ممکن است همان مدل برای مسیر پایه و مدل دوم برای استثناها مناسب باشد. معیار اصلی این است که سیستم ترکیبی بعد از Routing و Review چه کیفیتی دارد، نه اینکه یک مدل در جدول رتبهٔ بالاتری بگیرد.

نمونهٔ پیشنهادی برای داشبورد پذیرش: نرخ فیلد حساس اشتباهِ پذیرفته‌شده، درصد سند کامل و معتبر، نرخ Review، نرخ رد ورودی، هزینه به‌ازای سند پذیرفته‌شده و p95 زمان آماده‌سازی. در RAG، Recall شاهد و صحت Citation و پاسخ را اضافه کنید. نرخ Review را کنار دقت نشان دهید؛ یک سیستم که همهٔ موارد دشوار را رد می‌کند شاید روی موارد پذیرفته‌شده عالی باشد ولی برای محصول کاربرد کمی داشته باشد. مقدار آستانه را با هزینهٔ خطا و محدودیت تیم بازبینی تعیین کنید، نه صرفاً Confidence خام مدل.

در بحث‌های عمومی هم این تمایز دیده می‌شود: یک توسعه‌دهنده در r/LocalLLaMA دربارهٔ جایگزینی Granite-Docling با VLM دوردست و سازگاری DocTags سؤال کرده بود، در حالی که یکی از پاسخ‌ها به تفاوت دقت متن، ساختار جدول و فیلد نرمال‌شده اشاره داشت. قرارداد خروجی و سازگاری موتور را باید مثل یک رابط نرم‌افزاری آزمایش کنیم؛ مدل دیگری که Markdown تولید می‌کند لزوماً جایگزین مستقیم مدلی نیست که DocTags تولید می‌کند. [۳۷](#ref-37)

### از آزمایش محدود تا Regression Seed قابل‌تکرار

یک جدول نتیجه بدون ورودی و تنظیمات برای تصمیم Production کافی نیست. برای هر آزمایش، حتی یک Crop، یک Manifest (فهرست هویت و شرایط اجرا) بسازید. این Manifest باید شامل هش فایل و Crop، صفحه و مختصات، DPI و پیش‌پردازش، نسخهٔ Parser/Backend، مدل و Revision دقیق یا شناسهٔ Endpoint، Quantization، Runtime، سخت‌افزار، Prompt و Schema نسخه‌دار/هش‌دار، Reasoning Effort و محدودیت توکن (در صورت کاربرد)، مسیر Router، خروجی خام و مرجع تأییدشده، نسخهٔ Scorer، معیار و نتیجه باشد. برای API بسته، تنظیمات افشا نشده را «نامشخص» بنویسید؛ به‌جای نسخهٔ دقیق از نام خانواده استفاده نکنید.

برای شفافیت، وضعیت شواهد آزمایش‌های نقل‌شده در این مقاله را یک‌جا مشخص می‌کنم:

| آزمایش ثبت‌شده | اطلاعات موجود | چه چیزی برای بازتولید کامل کم داریم؟ |
|---|---|---|
| PyMuPDF / PUA | یک صفحه، ۳۰۸ از ۹۸۰ نویسه | فایل/هش مجاز، فونت و CMap، نسخهٔ دقیق و خروجی مرجع |
| Docling / TableFormer | مشاهدهٔ نقص ساختار، Unicode و فرمول؛ تنظیمات توصیف‌شده در بخش ۶ | فایل مجاز، نسخه و وزن، Runtime، تنظیم کامل و Gold ساختاری |
| Gemini 3 Flash | ۲۶ سلول مرجع، ۱۳ سلول اضافی و ۴۹ رخداد حساس | شناسهٔ Snapshot API، Crop/رزولوشن، Prompt، Schema و Gold/Prediction قابل انتشار |
| Bina Rizeh | ۳۷ Crop واقعی، چهار خانوادهٔ سند، Scorer ثابت و نتایج حساس | هش دقیق Checkpoint، فایل‌ها و برچسب‌ها، تنظیم کامل اجرا و محیط قابل انتشار |
| PageIndex Page Extract | استخراج هدایت‌شدهٔ چند سطر و فرمول | نسخهٔ سرویس، Schema/Prompt، ورودی و خروجی مرجع کامل |
| Inkling 2 Mini | غربال تصویر مصنوعی، ۳ از ۵ رقم و صفر از ۲ عبارت | Checkpoint/Endpoint، Runtime، تنظیم Reasoning/Quantization، Prompt و تصویر اصلی |

جزئیات بالا از سوابق در دسترس استخراج شده‌اند؛ تصویر اصلی و Manifest کامل این آزمایش‌ها برای انتشار در اختیار ندارم. بنابراین هیچ تصویر ساختگی را به‌عنوان Crop واقعی و هیچ پیش‌بینی ساختگی را به‌عنوان خروجی تاریخی منتشر نمی‌کنم. شکل‌های آموزشیِ این مقاله برای تعریف Regression Seed مفیدند، ولی نتیجهٔ اجرای مدل روی آن‌ها نیستند. برای نمونهٔ قابل اشتراک، یک جدول مصنوعی با سلول ۱۸٪ و یک خروجی فرضی ۸۱٪ طراحی کنید؛ مرجع و Prediction فرضی را جدا برچسب بزنید، و سپس همان نمونه را روی نسخه‌های واقعی اجرا و ثبت کنید. برای آزمایش عمومی واقعیِ دیگران، بستهٔ بازتولید Bina 0.4 را ببینید [۴۲](#ref-42).

### هزینه و کیفیت در مقیاس محصول

برای مقایسهٔ Parser و VLM مستقیم، قرارداد یکسان (پوشش، فیلد، ساختار، شاهد) تعریف کنید. هزینهٔ کل خوانش، Retry، زیرساخت، نگهداری، ذخیره، بازبینی و Re-index را بر تعداد رکورد یا سند واقعاً پذیرفته‌شده تقسیم کنید. **[مثال فرضی]** هزار صفحه با مسیر بصری ۰٫۰۲ واحد به ازای صفحه، ۲۰ واحد هزینهٔ خوانش دارد؛ مسیر پایه ۰٫۰۰۲ واحد و Fallback برای ۱۰٪ صفحات با همان قیمت بصری، ۴ واحد هزینهٔ خوانش دارد. اگر Router صفحهٔ حساس را جا بیندازد یا بازبینی و نگهداری گران شود، این تفاوت به تنهایی دلیل انتخاب نیست.

در مطالعهٔ فنی NVIDIA در ۲۰۲۵، روی A100، مسیر OCR تخصصی ۰٫۱۱۸ ثانیه به‌ازای صفحه و مرحلهٔ تصویر به متن Llama 3.2 Vision 11B برابر ۳٫۸۱ ثانیه به‌ازای صفحه گزارش شد. این دو زمان دامنهٔ یکسان ندارند؛ برای تصمیم‌گیری End-to-End هر دو مسیر را با خروجی و شرایط همسان اندازه بگیرید. در همان مطالعه، تفسیر نمودار بدون برچسب عددی یکی از موارد ارزشمند VLM بود. [۳۵](#ref-35) PureDocBench نیز رقابت مدل‌های کوچک تخصصی با مدل‌های عمومی و حساسیت به تخریب سند را گزارش می‌کند. [۳۶](#ref-36)

در محصول یک‌باره، مهندسی یک Router پیچیده ممکن است ارزش نداشته باشد؛ در سرویس پرتکرار، هزینهٔ ثابت توسعه در برابر هزینهٔ متغیر هر صفحه مهم می‌شود. برای تصمیم دربارهٔ **Full Context یا Retrieval**، Parsing را ثابت نگه دارید. Lost in the Middle و Context Rot نشان می‌دهند صرف وجود اطلاعات در Context به معنی استفادهٔ درست از آن نیست. گزارش *You don't need long context* (۳ نوامبر ۲۰۲۶) در وظایف بررسی‌شده با Compaction مکرر (فشرده‌سازی وضعیت و حفظ Context فعال کوتاه) افت ناچیز یا بدون افت دقت گزارش می‌کند؛ در سه مدل Open-weight روی پیکربندی‌های آزمایش‌شده، هزینهٔ تولید توکن در Context ۲۵۶K نسبت به ۱۶K تا ۷ برابر بیشتر بود. پیام پژوهش کنترل طول Context فعال است، نه حذف قطعی Long Context یا اثبات عملکرد RAG فارسی. [۲۱](#ref-21) روی پرسش‌های واقعی، Full Context، RAG و Hybrid را با Recall شاهد، صحت پاسخ/استناد، p95 و هزینهٔ Query بسنجید.

### پیاده‌سازی با پذیرش نهایی فرق دارد

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-24-title" direction="ltr" height="294" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 294" width="720" xml:lang="fa"><title id="v5-figure-24-title">سه دروازهٔ مستقل پذیرش</title><rect x="0" y="0" width="720" height="294" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#fffdf9" height="266" rx="8" width="720" x="0" y="0"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="700" text-anchor="middle" unicode-bidi="isolate" x="360" y="32">سه دروازهٔ مستقل پذیرش</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="52"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="82">تست نرم‌افزار: قراردادها و رفتارهای مشخص</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="116"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="146">اجرای تازه: خروجی واقعی، نه کش قدیمی</text><rect fill="#e1f1ef" height="49" rx="6" stroke="#aab8b5" width="670" x="25" y="180"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="17" text-anchor="middle" unicode-bidi="isolate" x="360" y="210">ارزیابی مرجع: کیفیت و پوشش روی اسناد نماینده</text></g></svg><figcaption>[مسیر پیشنهادی] تست سبز به تنهایی به معنی پذیرش داده نیست.</figcaption></figure>

Unit Test موفق نشان می‌دهد رفتارهای مشخصی را کنترل کرده‌ایم؛ اما رفع یک باگ روی نمونهٔ کوچک با عبور کل مجموعهٔ مرجع از معیار پذیرش فرق دارد. برای اعتماد به نسخهٔ جدید، اجرای تازه (نه Replay کش)، بررسی موارد شکست و اعتبارسنجی مستقل روی نمونه‌های نماینده لازم است. وضعیت پذیرش هر نسخه را جدا از وضعیت اجرای Job و نتیجهٔ تست نرم‌افزاری ثبت کنید.

## ۱۴. ابزارها و مدل‌ها؛ کدام برای کدام مسئولیت؟

این فهرست را به چند طبقه تقسیم کرده‌ام تا ابزار سطح پایین PDF، مدل OCR، مدل Layout، Parser کامل و VLM عمومی با هم اشتباه گرفته نشوند. ستون «شواهد» وضعیت آن‌ها را در همین مطلب مشخص می‌کند. «بررسی‌شده به‌عنوان گزینه» به معنی اجرای Benchmark موفق نیست. منابع رسمی، قابلیت‌های مستند ابزارها را توضیح می‌دهند؛ نتایج شخصیِ محدود را مستقلاً تأیید نمی‌کنند.

| ابزار / فناوری | لایه و کاربرد | شواهد و جهت بررسی |
|---|---|---|
| PyMuPDF | PDF Decoder، متن Native، رندر، هندسه و تشخیص جدول ساده | استفاده و آزمایش ثبت‌شده؛ برای منبع شواهد و Baseline، نه پذیرش خودکار متن |
| Docling | تبدیل و بازسازی ساختار سند با Pipeline قابل تنظیم | استفاده و آزمایش ثبت‌شده؛ کنترل مستقل Layout/OCR/جدول |
| TableFormer | مدل تشخیص ساختار جدول در مسیر Docling | آزمایش تنظیمات؛ ساختار ناقص در نمونهٔ دشوار، نه شکست قطعی روی همهٔ داده‌ها |
| Gemini 3 Flash | VLM عمومی برای خوانش یا استخراج هدایت‌شده | آزمایش محدود جدول با ۱۷/۲۶ تطابق دقیق و ۱۳ سلول اضافه |
| PageIndex Page Extract | استخراج ساختاریافته و هدایت‌شده | آزمایش محدود چند سطر و یک فرمول؛ برای رتبه‌بندی کافی نیست |
| Inkling 2 Mini | مدل بررسی‌شده در غربال ورودی تصویری | غربال اولیهٔ همان کاربرد ناموفق؛ توانایی بینایی عمومی استنتاج نمی‌شود |
| Bina Rizeh | OCR تخصصی فارسی | آزمایش ۳۷ Crop واقعی؛ معیارهای حساس ضعیف در همان بررسی |
| PaddleOCR / PP-OCR | Detection و Recognition متن؛ گزینه‌های چندزبانه | در فهرست گزینه‌ها؛ بررسی مدل و نسخهٔ دارای فارسی [۶](#ref-6) |
| PP-DocLayoutV3 / PP-StructureV3 | Layout و Pipeline بازسازی سند در اکوسیستم PaddleOCR | گزینهٔ بررسی؛ نسخه و تنظیمات دقیق، Coverage و Table Recognition را مقایسه کنید [۶](#ref-6) |
| PaddleOCR-VL / VL-1.5 / VL-1.6 | VLM تخصصی؛ مؤلفهٔ VL-1.6 با ۰٫۹ میلیارد پارامتر برای بازسازی عناصر سند؛ مسیر مدل‌محور در کنار PP-StructureV3 | گزینهٔ بررسی، نه آزمایش شخصی؛ کیفیت متن فارسی، جدول و پوشش را جداگانه بسنجید [۶](#ref-6) |
| Surya 2 / 2.5 | OCR، Layout و Reading Order؛ خروجی HTML جدول در 2.5 | گزینهٔ بررسی؛ 2.5 هندسهٔ سطر و ستون را مانند مسیر سادهٔ Surya 2 ارائه نمی‌کند؛ قرارداد خروجی نسخه را جدا بسنجید [۷](#ref-7) |
| Marker / MinerU | Pipelineهای سطح سند با خروجی ساختاریافته | گزینه‌های بررسی؛ Baseline End-to-End و مقایسهٔ اجزا [۸](#ref-8) |
| [LlamaParse](https://developers.llamaindex.ai/llamaparse/parse/) | سرویس Parsing سطح سند | گزینهٔ بررسی؛ کیفیت، قرارداد خروجی، هزینه و سیاست داده را بسنجید [۴۶](#ref-46) |
| pypdf / pdfplumber / Camelot | خواندن PDF و روش‌های متنی/هندسی جدول | گزینه‌های پژوهش عمومی؛ Baseline برای سند دیجیتال [۱](#ref-1) [۱۳](#ref-13) |
| Tesseract / OCRmyPDF | OCR و ساخت PDF قابل جست‌وجو | گزینه‌های پژوهش عمومی؛ ارزیابی زبان و کیفیت اسکن [۴](#ref-4) [۵](#ref-5) |
| Qwen2.5-VL و VLMهای مشابه | خوانش تصویری و تولید ساختاریافته | گزینهٔ پژوهش عمومی برای مسیر محلی؛ نیازمند Benchmark همان کاربرد [۱۲](#ref-12) |
| olmOCR | تبدیل محتوای PDF/تصویر به نمایش متنی مناسب LLM | گزینهٔ پژوهش عمومی برای مقایسهٔ End-to-End؛ شواهد کیفیت فارسی در این مقاله نداریم [۱۲](#ref-12) |

قبل از استقرار، مجوز، سخت‌افزار، هزینه، محرمانگی و امکانات نسخهٔ نصب‌شده را بررسی کنید.

### وضعیت گزینه‌های فارسی‌مرتبط در ۲۶ نوامبر ۲۰۲۶

این مرور، کاتالوگ گزینه‌های آزمایش است، نه Leaderboard و نه گزارش تجربهٔ شخصی جدید. نسخه‌های تازهٔ مدل‌ها را به نتایج تاریخی بخش ۸ یا ۱۰ نسبت نمی‌دهم. امکان استفادهٔ تجاری، دسترسی به وزن‌ها، نیازهای Runtime و نوع خروجی را از کارت همان نسخه بررسی کنید.

| گزینه | مسئولیت و تغییر مهم | برای محصول فارسی چه بسنجیم؟ |
|---|---|---|
| [Bina-0.2-Rizeh](https://huggingface.co/Reza2kn/Bina-0.2-Rizeh)، [RizehPizeh](https://huggingface.co/Reza2kn/Bina-0.2-RizehPizeh) | خانوادهٔ تشخیص فارسی؛ RizehPizeh مدل Recognition خط مبتنی بر PP-OCRv6 طبق کارت مدل | کیفیت خط، عدد، علامت و قرارداد ترتیب؛ نیاز به Detection/Grouping مستقل |
| [Bina-0.3-Koochik](https://huggingface.co/Reza2kn/Bina-0.3-Koochik)، [Bina-0.4-Koochik](https://huggingface.co/Reza2kn/Bina-0.4-Koochik)، [Bina-0.4-Rizeh](https://huggingface.co/Reza2kn/Bina-0.4-Rizeh) | نسخه‌های جدید خانواده؛ 0.3 مسیر صفحه و 0.4 گزینه‌های Recognition/مدل‌های محلی و بستهٔ ارزیابی عمومی دارد؛ نسخه‌های 0.1/0.2 نیز در تاریخچهٔ خانواده‌اند | تست تازه با مدل دقیق، دامنهٔ آموزش، مجوز مخصوص هر مدل، مقایسهٔ خط و صفحه و هزینه؛ نمره‌های توسعه را جای Gold مستقل نگذارید [۴۲](#ref-42) |
| [Persian Pixel](https://huggingface.co/datasets/Omarrran/Persian_Pixel) | منبع دادهٔ مصنوعی، نه مدل یا Parser؛ جفت تصویر و متن با تخریب کنترل‌شده | پوشش فونت و حالت‌های مرزی برای Training/Stress؛ صحت واقعی روی Gold دست‌نخورده [۳۸](#ref-38) |
| [DeepSeek-OCR Persian fine-tune (ParsAiGroup)](https://huggingface.co/ParsAiGroup/deepseek_ocr_hezar4k_persian) | نمونهٔ عمومی Fine-tune روی دادهٔ فارسی Hezar بر اساس توضیح کارت؛ هیچ تست شخصی در این مقاله ندارد | بهبود CER و حساسیت اعداد را روی دادهٔ خارج از آموزش نسبت به مدل پایه بسنجید؛ ادعای بهبود شدید بدون مقایسهٔ ثابت پذیرفته نیست [۴۴](#ref-44) |
| [dots.ocr](https://github.com/studio-dots-ai/dots.ocr) / [dots.mocr](https://github.com/studio-dots-ai/dots.mocr) | خانوادهٔ مدل‌محور؛ نام‌گذاری 1.5 در مارس ۲۰۲۶ تغییر کرده است | پوشش ناحیه، روابط، صحت خوانش؛ نام مدل دقیق و نسخهٔ کد را ثبت کنید [۱۸](#ref-18) |
| [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md) | مسیر مدل‌محور 0.9B همراه PP-DocLayoutV3؛ در Release 3.7.0 (۱۱ ژوئن ۲۰۲۶) عرضه شده است | مقایسه با VL-1.5 و PP-StructureV3؛ آیا هندسه و متن خام قابل استفاده‌اند؟ [۶](#ref-6) |
| [Surya 2.5](https://github.com/datalab-to/surya/releases) | طبق Release 0.23.0، مدل پیش‌فرض جدیدتر؛ جدول به HTML ولی بدون هندسهٔ سطر/ستون؛ مسیر simple به Surya 2 نیاز دارد | اگر TableIR هندسی لازم دارید، تغییر نسخه را به‌عنوان تغییر قرارداد خروجی آزمایش کنید، نه Upgrade بی‌خطر [۷](#ref-7) |
| [Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)، [2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0)، [Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) | مسیر تخصصی با خروجی متن، جدول و شواهد مکانی؛ نسخهٔ 3.5 در کارت مدل قابلیت نمودار، دسته‌بندی عناصر و ترتیب خواندن را ذکر می‌کند | پوشش کامل، تشخیص متن فارسی و رابطهٔ سلول‌ها، هزینهٔ Runtime؛ ارزیابی چندزبانهٔ سازنده تضمین فارسی نیست [۴۵](#ref-45) |

مدل کوچکِ تخصصی را به‌خاطر اندازهٔ کوچک یا مدل بزرگ را به‌خاطر هوشمندی عمومی انتخاب نکنید. Fine-tune ممکن است روی دادهٔ هم‌دامنه خطای نویسه را کم کند، ولی ممکن است روی فونت، مهر یا جدول جدید افت کند. مقایسهٔ مدل پایه و Fine-tune را با ورودی، Split، Scorer و قرارداد خروجی ثابت انجام دهید؛ اگر نتیجه فقط از روی X یا معرفی سازنده است، آن را سرنخ آزمایش بنامید نه نمرهٔ مستقل.

### بازنگری جدول‌های مقایسهٔ سال ۲۰۲۵؛ از «High Accuracy» به معیار قابل آزمایش

در اوت ۲۰۲۵ در کانال دو جدول مقایسهٔ OCR و PDF Parser منتشر کردم که سرعت، دقت، چندزبانه بودن، جدول و بازخورد کاربران را کنار هم قرار می‌دادند. آن جدول‌ها برای پیدا کردن ابزارهای جدید مفید بودند، ولی برای انتخاب Production کافی نیستند: «High»، «Fast» و «Excellent» تعریف مشترک ندارند و بازخوردهای عمومی بدون لینک، نسخه و نمونهٔ خطا قابل تکرار نیستند. اینجا همان گزینه‌ها را بازآرایی می‌کنم تا خواننده بداند هر کدام چه مسئله‌ای را حل می‌کنند و چه چیزی را باید آزمایش کند.

#### مدل‌ها و ابزارهای خوانش تصویری و Parsing مدل‌محور

| ابزار | نقش و خروجی مستند | چه چیزی را دربارهٔ سرعت می‌دانیم؟ | سؤال اصلی روی دادهٔ فارسی |
|---|---|---|---|
| [dots.ocr](https://github.com/studio-dots-ai/dots.ocr) / [dots.mocr](https://github.com/studio-dots-ai/dots.mocr) | مدل چندزبانهٔ ترکیب Layout، خوانش و ترتیب عناصر؛ نسخهٔ اولیه با ۱٫۷ میلیارد پارامتر؛ نسخهٔ dots.ocr-1.5 در ۱۹ مارس ۲۰۲۶ به dots.mocr تغییر نام داد | کوچک‌تر بودن مدل نسبت به بعضی رقبا، تضمین سرعت کل Pipeline نیست | پوشش ناحیه‌ها، صحت RTL، جدول و اعداد فارسی |
| [olmOCR](https://github.com/allenai/olmocr) | ابزار تبدیل صفحات به متن خوانا برای LLM؛ مجموعهٔ ارزیابی olmOCR-Bench | عدد ۰٫۸۴ صفحه‌برثانیه روی H100 در جدول قدیمی آمده بود، اما روش اندازه‌گیری مشترک و قابل تأیید در اختیار نداریم | جدول، فرمول، ترتیب خواندن و Hallucination؛ گزارش کاربران فقط جهت آزمون است |
| [RolmOCR](https://huggingface.co/reducto/RolmOCR) | جایگزین معرفی‌شده توسط Reducto برای olmOCR | توسعه‌دهنده هدف بهبود سرعت و مصرف حافظه را گزارش کرده؛ روی سخت‌افزار و دادهٔ یکسان بسنجید | آیا کاهش هزینه با افت پوشش یا دقت فیلد همراه است؟ |
| [OCRFlux](https://github.com/chatdoc-com/OCRFlux) | تبدیل PDF به Markdown و مسیر ادغام محتوای ادامه‌دار و جدول چندصفحه‌ای؛ مدل OCRFlux-3B | عدد ۲۵ صفحه‌برثانیه روی H100 در جدول قدیمی آمده بود؛ بدون شرایط بازتولید، آن را مبنای تخمین نمی‌گذارم | اتصال درست جدول‌های ادامه‌دار و جلوگیری از ادغام جدول‌های نامرتبط |
| [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR) | رویکرد ساختار، شناخت محتوا و روابط؛ نسخه‌های منتشرشده مانند 3B و pro-1.2B | نسخهٔ ۲۵۶ میلیونی در جدول قدیمی به‌اشتباه به این ابزار نسبت داده شده بود | کیفیت Recognition و روابط روی فارسی؛ نتایج انگلیسی/چینی جای تست فارسی نیست |
| [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) | مدل تصویر به Markdown، از جمله بازنمایی فرمول و عناصر ساختاری | رتبه و سرعت عمومی بدون تنظیمات مشترک نامشخص است | وفاداری متن و ساختار، حذف یا افزودن محتوا، صحت فرمول |

کارت مدل و مستندات نسخهٔ منتخب را برای جزئیات هر ردیف ببینید. [۱۸](#ref-18)

#### Parserها، مدل‌های تخصصی و ابزارهای تبدیل فرمت

| ابزار | نوع و کاربرد | جهت ارزیابی برای انتخاب |
|---|---|---|
| Marker | Pipeline سطح سند برای بازسازی و تبدیل به خروجی خوانا | پوشش، جدول، ترتیب عناصر و زمان کل تبدیل؛ نه صرف سرعت مدل |
| Docling | Framework قابل تنظیم با نمایش ساختاریافته و مدل‌های قابل تعویض | امکان کنترل مؤلفه‌ها و بازیابی ناحیه‌ای؛ زمان کل و فارسی را بسنجید |
| SmolDocling-256M-preview | مدل تخصصی صفحه به DocTags، نه Framework کامل | کارت مدل میانگین ۰٫۳۵ ثانیه به‌ازای صفحه روی A100 با vLLM گزارش کرده؛ برای GPU معمولی یا کل Pipeline تضمین نیست |
| Granite-Docling-258M | مدل تخصصی تولید DocTags و تبدیل به DoclingDocument | مقایسه با SmolDocling و مسیر کلاسیک؛ پشتیبانی آزمایشی عربی معادل کیفیت فارسی نیست |
| [Mathpix](https://docs.mathpix.com/) | سرویس خوانش محتوای علمی، ریاضی، دست‌خط و جدول | صحت فرمول، LaTeX و هزینهٔ API؛ Nanonets-OCR-s محصول جداگانهٔ Nanonets است، نه زیرمجموعهٔ Mathpix |
| [LlamaParse](https://developers.llamaindex.ai/llamaparse/parse/) | سرویس پردازش اسناد برای کاربردهایی مانند RAG | خروجی جدول و Layout، ردگیری منبع، محدودیت حجم و هزینه [۴۶](#ref-46) |
| [MarkItDown](https://github.com/microsoft/markitdown) | ابزار سبک تبدیل فرمت‌های متنوع به Markdown | Baseline مناسب برای ورودی‌هایی که متن خوانا کافی است؛ حفظ کامل روابط و Provenance را پیش‌فرض ندانید |

دربارهٔ SmolDocling، مقالهٔ اصلی رقابت با مدل‌هایی تا ۲۷ برابر بزرگ‌تر را در آزمایش‌های مشخص گزارش کرده است؛ این به معنی شکست دادن همهٔ آن مدل‌ها در همهٔ وظایف نیست. همچنین اندازهٔ فایل وزن‌ها با Peak VRAM شامل فعال‌سازی، Context و Batch یکسان نیست؛ عدد کمتر از ۵۰۰ مگابایت در پست قدیمی را به‌عنوان بودجهٔ قطعی حافظه استفاده نکنید. [۱۹](#ref-19)

از تجربهٔ کامیونیتی می‌توان سؤال آزمایش ساخت: مثلاً دربارهٔ جدول‌های پیچیده و Hallucination در olmOCR یا کند بودن Docling در بعضی تست‌های من. اما قبل از ثبت نتیجه، نسخهٔ مدل، Backend، دستگاه، رزولوشن، طول خروجی، Batch Size و Cold/Warm بودن اجرا را ثابت و مستند کنید. نرخ Throughput گروهی (صفحه بر ثانیه) با Latency تک‌درخواست و p95 زمان انتهابه‌انتها یکی نیست. به‌جای یک ستون مبهم «دقت»، حداقل CER، صحت ساختار جدول، Recall ناحیه‌ها، دقت فیلدهای حساس و نرخ نیاز به بازبینی را ثبت کنید. اگر ابزار از زبان‌های مختلف پشتیبانی می‌کند، باز هم روی نمونه‌های فارسی آزمایش بگیرید؛ Multilingual یک قابلیت ادعایی است نه یک نمرهٔ فارسی.

## ۱۵. Backend و امنیت؛ از Parser به سرویس قابل‌نگهداری

### ذخیرهٔ پاسخ خام قبل از Validation

در یک رخداد واقعی، مدل پاسخی شامل ۳۲ ناحیهٔ پردازش‌شده تولید کرد، اما مرحلهٔ Validation دچار خطا شد و پاسخ استخراج‌شده از دست رفت. تعویض مدل این مشکل را حل نمی‌کرد. لازم بود Artifact خام قبل از بررسی و تبدیل، به‌شکل پایدار ذخیره شود تا بتوان خروجی را بعداً بررسی یا Replay کرد.

این کار در Pipelineهای پرهزینه ارزش زیادی دارد. اگر مدل خروجی تولید کرده و فقط مرحلهٔ ساخت IR شکست خورده، تکرار کل درخواست API ممکن است هم هزینهٔ اضافی ایجاد کند و هم نتیجهٔ متفاوتی بدهد. البته Replay باید صریح برچسب بخورد؛ نمی‌توان آن را آزمایش تازهٔ مدل گزارش کرد.

### Cache، Fresh Run و Rebuild را از هم جدا کنید

در یک بررسی، مشخص شد اجرای Parser با `force=True` الزاماً به معنی استخراج مجدد نبود؛ امکان داشت پاسخ قبلی از Cache خوانده و فقط خروجی مشتق‌شده بازسازی شود. اگر همان زمان مدل OCR را عوض کرده باشید، ممکن است به اشتباه تصور کنید مدل جدید کیفیت بهتری دارد.

چهار عملیات مستقل برای چنین سیستمی مفیدند:

| عملیات | مسئولیت |
|---|---|
| Verify-only | بررسی و اعتبارسنجی خروجی موجود، بدون استخراج دوباره |
| Rebuild-derived | ساخت مجدد Markdown، HTML یا Chunk از IR معتبر قبلی |
| Reparse-regions | استخراج تازهٔ ناحیه‌های مشخص و ثبت نسخهٔ جدید |
| Reparse-document-fresh | استخراج واقعی مجدد کل سند با کنترل صریح Cache |

کلید Cache باید هویت محتوا (نه صرفاً نام فایل)، نسخهٔ مؤلفه، تنظیمات، مدل و Prompt، پارامترهای رندر و Schema مربوط را در بر بگیرد. وابستگی بین مراحل را ثبت کنید. اگر فقط Markdown Exporter عوض شده، اجرای دوبارهٔ OCR منطقی نیست؛ ولی اگر رندر تصویر یا Recognizer تغییر کند، خروجی‌های وابسته نیاز به بررسی مجدد دارند.

### Job، Retry و Idempotency

اگر Worker وسط پردازش Crash کند یا دو کاربر هم‌زمان همان سند را آپلود کنند، چه اتفاقی می‌افتد؟ هر عملیات مهم باید وضعیت، نسخه و Artifact مشخص داشته باشد. Idempotency یعنی تکرار یک عملیات با هویت یکسان نباید اثر جانبی تکراری و ناخواسته ایجاد کند. برای کارهای طولانی Retry محدود، Timeout، مدیریت هم‌زمانی و امکان بازیابی از آخرین مرحلهٔ ثبت‌شده را در نظر بگیرید.

موفقیت Job را با پذیرش محتوای سند یکی نکنید. حتی اگر همهٔ درخواست‌های API پاسخ ۲۰۰ داده باشند، هنوز ممکن است ناحیه‌ای جا افتاده باشد. شمارش ورودی‌ها، صفحه‌ها، ناحیه‌ها و خروجی‌های معتبر و ثبت وضعیت هر بخش جلوی این اشتباه را می‌گیرد.

### مشاهده‌پذیری و هویت هر رکورد پذیرفته‌شده

برای هر رکوردی که اجازهٔ ورود به محصول می‌گیرد، صرفاً `model=Gemini` یا `ocr=Paddle` کافی نیست. یک Run ID و Artifact/Record ID پایدار به آن وصل کنید: هش سند و Crop، صفحه/ناحیه، نسخهٔ Decoder و Layout/OCR، شناسهٔ مدل و Revision یا Endpoint، Quantization و Runtime، DPI/تبدیل مختصات، نسخه و هش Prompt/Schema، علت و تصمیم Router، مسیرهای Fallback، نسخهٔ Validator و مرجع ارزیابی، نتیجهٔ بررسی و زمان/هزینه. اگر تنظیمات سرویس افشا نشده‌اند، مقدار نامعلوم را ثبت کنید. در UI بازبینی باید بتوان از رکورد پذیرفته‌شده به شاهد و خوانش‌های Candidate برگشت. اگر نسخهٔ مدل، Prompt یا Router تغییر کرد، رکورد قبلی را بی‌صدا بازنویسی نکنید؛ نسخهٔ جدید و وابستگی‌ها را جدا ثبت کنید.

در Dashboard عملیاتی، نرخ شکست و Timeout، پوشش، Review/Quarantine، نرخ پذیرش غلط روی نمونهٔ ممیزی‌شده، هزینهٔ هر رکورد پذیرفته‌شده، زمان صف و p95 اجرا را به تفکیک خانوادهٔ سند، نسخهٔ مدل و Route نشان دهید. تغییر توزیع فونت، اسکن یا قالب می‌تواند کیفیت را تغییر دهد حتی اگر API سالم باشد. Log و Trace نباید متن محرمانه، کلید API یا تصویر خصوصی را بی‌دلیل ذخیره کنند؛ فقط شناسه‌های لازم و Artifactهای کنترل‌شده را ثبت کنید.

### امنیت، دادهٔ محرمانه و اتصال به RAG

ورودی‌های PDF و Office را غیرقابل اعتماد فرض کنید. برای نوع و اندازهٔ واقعی فایل، تعداد صفحه، رزولوشن، حافظه، زمان و تعداد اجرای هم‌زمان محدودیت تعریف کنید. Worker پردازش فایل را از سرویس اصلی جدا کنید و حداقل دسترسی لازم را بدهید. فایل خام، Artifact و خروجی نباید به‌صورت پیش‌فرض عمومی شوند. قبل از ارسال سند به API خارجی، مجوز و سیاست نگهداری داده را بررسی کنید. محتوای سند هم برای Agent صرفاً داده است؛ دستورهای داخل سند نباید به‌عنوان دستور سیستم اجرا شوند. [۱۴](#ref-14)

### مرز سرویس Parsing و مصرف‌کنندهٔ داده

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-23-title" direction="ltr" height="381" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 381" width="720" xml:lang="fa"><title id="v5-figure-23-title">مرز Parsing و راهبردهای بازیابی</title><rect x="0" y="0" width="720" height="381" fill="#fffdf9"/><g transform="translate(0 14)"><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="21" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="29">یک نمایش معتبر، چند مسیر بازیابی</text><rect fill="#e1f1ef" height="64" rx="10" stroke="#176e73" width="400" x="160" y="48"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="20" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="75">DocumentIR / TableIR پذیرفته‌شده</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="15" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="98">صفحهٔ منبع، ناحیه و روابط</text><path d="M360 112V137H120V158M360 137V158M360 137H600V158" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="72" rx="10" stroke="#aab8b5" width="220" x="10" y="158"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="120" y="186">Context کامل</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="120" y="210">کل سند</text><rect fill="#fffdf9" height="72" rx="10" stroke="#aab8b5" width="220" x="250" y="158"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="186">RAG ساختاری</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="210">بخش یا گروه ردیف</text><rect fill="#fffdf9" height="72" rx="10" stroke="#aab8b5" width="220" x="490" y="158"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="600" y="186">درختی / ترکیبی</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="600" y="210">گسترش هنگام نیاز</text><path d="M120 230V254H360M360 230V275M600 230V254H360" fill="none" stroke="#176e73" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/><rect fill="#fffdf9" height="62" rx="10" stroke="#aab8b5" width="400" x="160" y="275"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="19" font-weight="650" text-anchor="middle" unicode-bidi="isolate" x="360" y="301">پاسخ با شاهد اصلی</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="400" text-anchor="middle" unicode-bidi="isolate" x="360" y="324">سنجش صحت، استناد، هزینه و تأخیر</text></g></svg><figcaption>یک نمایش پذیرفته‌شده، چند مسیر بازیابی؛ متن منبع و رابطه‌ها ثابت می‌مانند، ولی نحوهٔ انتخاب Context با کاربرد تغییر می‌کند.</figcaption></figure>

Parser باید عناصر، ترتیب، روابط و شاهد قابل پیگیری را تحویل دهد. Chunking و Retrieval مصرف‌کنندهٔ این نمایش‌اند، نه مسئول بازسازی اطلاعاتی که Parser حذف کرده است. عنوان والد و شمارهٔ بند را با پاراگراف، سرستون و واحد را با گروه ردیف‌های جدول، و Caption را با شکل نگه دارید. اندازهٔ توکنی Chunk را بعد از تعیین مرز معنایی و با آزمون Retrieval انتخاب کنید. HybridChunker در Docling یک گزینه برای این مرحله است. [۲۰](#ref-20)

برای Full Context، RAG ساختاری یا Hybrid، همان دادهٔ پذیرفته‌شده را به کار ببرید و کیفیت بازیابی و پاسخ را جدا بسنجید؛ معیارهای بخش ۱۳ را ببینید. PageIndex برای بازیابی درختی را با Page Extract برای استخراج فیلد یکی ندانید. [۲۲](#ref-22) تکنیک‌های [Multi-Representation Search](https://qdrant.tech/documentation/tutorials-search-engineering/multi-representation-search/) و [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) می‌توانند به بازیابی کمک کنند، ولی Context تولیدشده شاهد اصلی نیست؛ آن را جدا ذخیره و به متن واقعی ارجاع دهید. [۲۳](#ref-23) [۲۴](#ref-24)

## ۱۶. معماری را چطور مرحله‌به‌مرحله تکامل بدهیم؟

اگر تازه شروع کرده‌اید، پیشنهاد نمی‌کنم در روز اول یک سیستم با ده مدل و چند Recovery Layer بسازید. مسیر زیر یک چارچوب تصمیم‌گیری است، نه معماری ثابت برای همه.

<figure class="visual"><svg xmlns="http://www.w3.org/2000/svg" aria-labelledby="v5-figure-25-title" direction="ltr" height="338" role="img" style="font-family: Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif; background: #fffdf9; color: #202323;" viewBox="0 0 720 338" width="720" xml:lang="fa"><title id="v5-figure-25-title">معماری پیشنهادی مسیریابی تطبیقی در سطح صفحه یا ناحیه؛ هنوز آزموده نشده است.</title><defs><marker id="arrow-25" markerHeight="7" markerUnits="strokeWidth" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0 0 L7 3.5 L0 7 Z" fill="#176e73"/></marker></defs><rect x="0" y="0" width="720" height="338" fill="#fffdf9"/><g transform="translate(0 14)"><rect fill="#f6f5f0" height="42" rx="8" stroke="#aab8b5" width="300" x="210" y="8"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="35">صفحهٔ ورودی</text><path d="M360 50L360 70" fill="none" marker-end="url(#arrow-25)" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="42" rx="8" stroke="#176e73" width="330" x="195" y="70"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="96">تحلیل صفحه یا ناحیه</text><path d="M360 112V132H120V150" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="58" rx="8" stroke="#aab8b5" width="220" x="10" y="150"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="120" y="175">متن Native</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="120" y="196">دیجیتال</text><path d="M120 208V230H360V248" fill="none" stroke="#176e73" stroke-width="2"/><path d="M360 112V132H360V150" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="58" rx="8" stroke="#aab8b5" width="220" x="250" y="150"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="175">VLM-first</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="196">اسکن فارسی</text><path d="M360 208V230H360V248" fill="none" stroke="#176e73" stroke-width="2"/><path d="M360 112V132H600V150" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#f6f5f0" height="58" rx="8" stroke="#aab8b5" width="220" x="490" y="150"/><text direction="ltr" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="600" y="175">VLM / OCR</text><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="14" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="600" y="196">ناحیهٔ پیچیده / مقایسه</text><path d="M600 208V230H360V248" fill="none" stroke="#176e73" stroke-width="2"/><rect fill="#e1f1ef" height="46" rx="8" stroke="#176e73" width="400" x="160" y="248"/><text direction="rtl" fill="#202323" font-family="Noto Sans Arabic, Tahoma, DejaVu Sans, sans-serif" font-size="18" font-weight="500" text-anchor="middle" unicode-bidi="isolate" x="360" y="276">بازسازی و بررسی</text></g></svg><figcaption>معماری پیشنهادی مسیریابی تطبیقی در سطح صفحه یا ناحیه؛ [مسیر پیشنهادی] مسیر پیشنهادی برای فارسی حساس: VLM-first برای خوانش بصری؛ OCR تخصصی برای مقایسه یا کاربرد تأییدشده.</figcaption></figure>

| سطح | چه چیزی بسازیم؟ | چه زمانی به مرحلهٔ بعد برویم؟ |
|---|---|---|
| ۱. Baseline دیجیتال | Native Extraction، شمارهٔ صفحه، شواهد هندسی و متن خام | اگر صفحات تصویری یا ساختار پیچیده به‌طور معنادار جا می‌افتند |
| ۲. Layout-aware | تشخیص ناحیه، Reading Order، جدول و IR ساده | اگر خطاهای متن اسکن و ساختار پیچیده باقی می‌مانند |
| ۳. چندمسیره و چندزبانه | VLM-first برای تصاویر فارسی، OCR تخصصی برای مقایسه، پیش‌پردازش، Routing و نرمال‌سازی فارسی | اگر نواحی حساس هنوز با روش‌های موجود خوانده نمی‌شوند |
| ۴. بازیابی انتخابی | Crop دقیق، موتور جایگزین یا VLM، ثبت تعارض و بازبینی | فقط اگر کاهش خطا در ارزیابی، هزینه و پیچیدگی را توجیه کند |
| ۵. Production | نسخه‌بندی، Provenance، Quality Gate، Cache، Job و انتشار هماهنگ | پذیرش بر اساس پوشش، کیفیت و ریسک واقعی کاربرد |

در هر سطح از همان ابتدا شواهد، تست و محدودیت‌ها را حفظ کنید؛ لازم نیست قابلیت اعتماد را تا مرحلهٔ پنجم عقب بیندازید. برای یک آرشیو متنی ساده شاید سطح اول یا دوم کافی باشد. برای اسناد حساس فارسی ممکن است به ترکیب بیشتری نیاز داشته باشید. هدف، استفاده از کمترین پیچیدگی‌ای است که به کیفیت موردنیاز می‌رسد.

### ایده‌هایی که ارزش آزمایش دارند، نه نسخه‌های قطعی

- **Adaptive Routing:** انتخاب Native، OCR یا VLM بر اساس نوع و کیفیت ناحیه. سود احتمالی، کاهش پردازش اضافی است؛ ریسک آن جا انداختن ناحیه توسط Router است. با مقایسهٔ پوشش، هزینه و نرخ پذیرش غلط ارزیابی کنید.
- **Multi-Engine Consensus:** استفاده از اختلاف خوانش‌ها برای کشف تعارض. بررسی کنید چند خطای واقعی پیدا می‌شود و چند بازبینی غیرضروری ایجاد می‌شود؛ رأی اکثریت را حقیقت تلقی نکنید.
- **Adaptive DPI:** افزایش رزولوشن فقط برای قسمت‌های دشوار. کیفیت نویسهٔ ریز و هزینهٔ رندر/حافظه را هم‌زمان بسنجید.
- **Cell-level Verification:** وضعیت مستقل برای سلول‌ها؛ همراه با بررسی روابط ساختاری کل جدول. صحت تک‌تک مقدارها بدون صحت ارتباط‌ها کافی نیست.
- **Synthetic Persian Dataset:** تولید کنترل‌شدهٔ جدول، فرم، ارقام و ترکیب راست‌به‌چپ با پاسخ معلوم. برای مطالعهٔ روش تولید داده، پیش‌چاپ Persian Pixel نمونهٔ یک مجموعهٔ مصنوعی فارسی با انواع تخریب تصویری است. دادهٔ مصنوعی برای تمرین و تست حالت مرزی مفید است، ولی کیفیت روی اسناد واقعی، جدول و دست‌خط را جدا بسنجید. [۳۸](#ref-38)
- **Human-in-the-loop:** نمایش هم‌زمان منبع و ساختار استخراج‌شده، اصلاح همراه با دلیل و ذخیرهٔ بازبینی برای Gold Dataset. هزینهٔ بازبینی و اختلاف برچسب‌زن‌ها را هم ثبت کنید.
- **Parsing به‌عنوان سرویس مستقل:** ورودی‌های مختلف و خروجی ساختاریافته با قرارداد نسخه‌دار برای RAG و Agent. مزیت چنین سرویسی می‌تواند از کیفیت و قابلیت پیگیری کل Pipeline بیاید، نه الزاماً یک مدل اختصاصی. این یک جهت محصولی پیشنهادی است، نه ادعای وجود محصول آماده.

## جمع‌بندی؛ از اینجا به بعد چه کار کنیم؟

اگر قرار باشد از این مقاله فقط یک روش برای حل مسئله با خودتان ببرید، پیشنهاد من این است: وقتی نتیجهٔ Parsing خراب است، قبل از انتخاب مدل جدید بپرسید **اولین مرحله‌ای که شواهد در آن خراب یا گم شده‌اند کجاست؟**

یک صفحهٔ مشکل‌دار پیدا کنید، تصویر اصلی و خروجی‌های میانی را کنار هم بگذارید، نوع خطا را مشخص کنید و یک تغییر محدود را آزمایش کنید. معیارش را هم از قبل تعیین کنید. بعد نتیجه را روی مجموعهٔ مرجع و نمونه‌های واقعی دیگر بسنجید. اگر تغییر فقط یک نمونه را درست کرد و ده نمونهٔ دیگر را خراب کرد، هنوز راهکار مناسبی نداریم.

برای سند فارسی به‌خصوص، تفاوت شکل ظاهری و Unicode، ترتیب نمایش و ترتیب منطقی، Recognition و بازسازی ساختار، مقدار و واحد و تعلق سلول به سرستون را همیشه در نظر بگیرید. یک مدل فارسی قوی می‌تواند کمک کند، ولی مسئولیت همهٔ این لایه‌ها را به‌تنهایی بر عهده ندارد. جمع‌بندی معماری من برای فارسی حساس این است: Native-first برای متن دیجیتال معتبر، VLM-first برای خوانش اسکن و عکس، و مقایسهٔ VLM مستقیم و مسیر ترکیبی برای محصولاتی که خروجی فیلدمحور و ساختار بصری می‌خواهند. مدل دستورپذیر و Schema بخش مهمی از سادگی این مسیرند؛ شواهد و پذیرش مستقل بخش مهمی از قابلیت اعتماد آن.

مهندسی Document Parsing در نهایت ترکیبی از مهندسی AI، شناخت فرمت سند، پردازش تصویر، ساختار داده، ارزیابی و مهندسی Backend است. مهم نیست چند مدل در Pipeline دارید؛ مهم این است که بدانید هر کدام چرا آنجا هستند، چه خطایی را حل می‌کنند، چطور ارزششان را اندازه گرفته‌اید و اگر خطا کردند، چه چیزی مانع ورود دادهٔ غلط به سیستم بعدی می‌شود.

## منابع و مطالعهٔ بیشتر

منابع رسمی برای توضیح قابلیت‌ها، پژوهش‌ها برای روش ارزیابی و گزارش‌های جامعه برای شناخت مسئله استفاده شده‌اند. نتایج شخصیِ برچسب‌خورده از آزمایش‌های ثبت‌شدهٔ محدود نویسنده‌اند و منابع خارجی تأیید مستقل آن‌ها نیستند. نمونهٔ گفت‌وگوها هدفمند و ناشناس است، نه نظرسنجی بازار. [کانال LLMEngineers](https://t.me/LLMEngineers) و [گروه AI/LLMs](https://t.me/AI_LLMs).

<a id="ref-1"></a>

۱. [pypdf — Why Text Extraction is Hard](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)، [PyMuPDF — Text recipes](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)

<a id="ref-2"></a>

۲. [PyMuPDF — Common Issues](https://pymupdf.readthedocs.io/en/latest/recipes-common-issues-and-their-solutions.html)، [Page API](https://pymupdf.readthedocs.io/en/latest/page.html)، [Text Extraction Details](https://pymupdf.readthedocs.io/en/latest/app1.html)

<a id="ref-3"></a>

۳. [Docling — Pipeline Options](https://docling-project.github.io/docling/reference/pipeline_options/)، [Advanced Options](https://docling-project.github.io/docling/usage/advanced_options/)، [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/)، [Model Catalog](https://docling-project.github.io/docling/usage/model_catalog/)

<a id="ref-4"></a>

۴. [Tesseract — Improving OCR Quality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)، [Language Data](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)

<a id="ref-5"></a>

۵. [OCRmyPDF — Cookbook](https://ocrmypdf.readthedocs.io/en/stable/cookbook.html)

<a id="ref-6"></a>

۶. [PaddleOCR — PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)، [PaddleOCR-VL-1.5](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.en.md)، [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md)، [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases)، [PP-OCRv5 multilingual models](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)

<a id="ref-7"></a>

۷. [Surya — Official repository](https://github.com/datalab-to/surya)، [Release notes — Surya 2.5 / v0.23.0](https://github.com/datalab-to/surya/releases)

<a id="ref-8"></a>

۸. [Marker](https://github.com/datalab-to/marker)، [MinerU](https://github.com/opendatalab/MinerU)

<a id="ref-9"></a>

۹. [Unicode UAX #9 — Bidirectional Algorithm](https://www.unicode.org/reports/tr9/)، [UAX #15 — Unicode Normalization Forms](https://www.unicode.org/reports/tr15/)

<a id="ref-10"></a>

۱۰. [OmniDocBench — Evaluation and annotations](https://github.com/opendatalab/OmniDocBench)

<a id="ref-11"></a>

۱۱. [OHR-Bench — OCR Hinders RAG](https://github.com/opendatalab/OHR-Bench)

<a id="ref-12"></a>

۱۲. [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)، [olmOCR](https://github.com/allenai/olmocr)

<a id="ref-13"></a>

۱۳. [pdfplumber](https://github.com/jsvine/pdfplumber)، [Camelot documentation](https://camelot-py.readthedocs.io/en/latest/)

<a id="ref-14"></a>

۱۴. [OCRmyPDF — PDF security](https://ocrmypdf.readthedocs.io/en/stable/pdfsecurity.html)

<a id="ref-15"></a>

۱۵. [PyMuPDF — Functions: Page.get_texttrace](https://pymupdf.readthedocs.io/en/latest/functions.html#Page.get_texttrace)، [PDF Association — PDF standards](https://pdfa.org/resource/iso-32000-2/)

<a id="ref-16"></a>

۱۶. [Granite-Docling-258M — Model card](https://huggingface.co/ibm-granite/granite-docling-258M)، [دموی رسمی](https://huggingface.co/spaces/ibm-granite/granite-docling-258m-demo)

<a id="ref-17"></a>

۱۷. [Google Document AI — Gemini Layout Parser](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk)، [Gemini API — Document understanding](https://ai.google.dev/gemini-api/docs/document-processing)

<a id="ref-18"></a>

۱۸. [dots.ocr](https://github.com/studio-dots-ai/dots.ocr)، [dots.mocr](https://github.com/studio-dots-ai/dots.mocr)، [RolmOCR — Model card](https://huggingface.co/reducto/RolmOCR)، [OCRFlux](https://github.com/chatdoc-com/OCRFlux)، [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)، [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s)

<a id="ref-19"></a>

۱۹. [SmolDocling — paper](https://arxiv.org/abs/2503.11576)، [SmolDocling-256M-preview — Model card](https://huggingface.co/docling-project/SmolDocling-256M-preview)، [Mathpix Docs](https://docs.mathpix.com/)، [MarkItDown](https://github.com/microsoft/markitdown)

<a id="ref-20"></a>

۲۰. [Docling — Chunking](https://docling-project.github.io/docling/concepts/chunking/)، [Hybrid chunking example](https://docling-project.github.io/docling/_generated/examples/hybrid_chunking/)

<a id="ref-21"></a>

۲۱. [Lost in the Middle (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/)، [Context Rot — Chroma](https://www.trychroma.com/research/context-rot)، [You don't need long context — technical report, November 2026](https://arxiv.org/abs/2611.05306)

<a id="ref-22"></a>

۲۲. [PageIndex — documentation](https://pageindex.ai/docs)، [PageIndex — Reasoning-based RAG](https://pageindex.ai/blog/pageindex-intro)

<a id="ref-23"></a>

۲۳. [Qdrant — Multi-Representation Search](https://qdrant.tech/documentation/tutorials-search-engineering/multi-representation-search/)

<a id="ref-24"></a>

۲۴. [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

<a id="ref-25"></a>

۲۵. [r/LocalLLaMA — پردازش فرم‌های صنعتی، جدول‌های تو‌در‌تو و محدودیت CPU](https://www.reddit.com/r/LocalLLaMA/comments/1s22bzm/hitting_a_wall_parsing_1000_complex_scanned_pdfs/)؛ گزارش کاربر، نه Benchmark مستقل

<a id="ref-26"></a>

۲۶. [r/RAG — اسناد مالی محرمانه و Trade-off کیفیت و Privacy](https://www.reddit.com/r/Rag/comments/1shdula/rag_for_complex_pdfs_struggling_with_parsing_vs/)؛ تجربهٔ فردی

<a id="ref-27"></a>

۲۷. [r/LangChain — مشکلات جدول، ستون‌بندی و Parser در RAG](https://www.reddit.com/r/LangChain/comments/1syu7vg/pdf_parsing_for_rag_is_still_a_mess_in_2026_whats/)؛ بحث جامعهٔ توسعه‌دهندگان

<a id="ref-28"></a>

۲۸. [Hacker News — PDF to Text, a challenging problem](https://news.ycombinator.com/item?id=43973721)؛ شامل تجربه‌ها و دیدگاه‌های سازندگان محصولات Document AI؛ برخی ذی‌نفع تجاری‌اند

<a id="ref-29"></a>

۲۹. [Amazon Textract — Features](https://aws.amazon.com/textract/features/) و [Azure Document Intelligence — Transparency note](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/document-intelligence/transparency-note)؛ برای شناخت قابلیت و قرارداد محصول، نه اثبات کیفیت فارسی

<a id="ref-30"></a>

۳۰. [Gemini API — Document understanding](https://ai.google.dev/gemini-api/docs/document-processing)، [Claude API — PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support)، [OpenAI API — File inputs](https://developers.openai.com/api/docs/guides/file-inputs)، [OpenAI — Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

<a id="ref-31"></a>

۳۱. [Docling — OCR](https://github.com/docling-project/docling/blob/main/docs/concepts/OCR.md)، [PyMuPDF — OCR](https://pymupdf.readthedocs.io/en/latest/ocr/index.html)؛ رفتار نسخهٔ نصب‌شده و زبان‌ها را جدا بررسی کنید

<a id="ref-32"></a>

۳۲. [Stack Overflow — RTL PDF text extraction](https://stackoverflow.com/questions/51511345/is-there-any-python-package-for-extracting-text-nicely-from-pdfs-in-rtl-language/75100290)، [Docling Issue #1938 — Arabic reversed text](https://github.com/docling-project/docling/issues/1938)، [r/learnpython — RTL PDF parser](https://www.reddit.com/r/learnpython/comments/1vrmrpr/pdf_parser_with_rtl_languages_support/)؛ گزارش‌های مشکل، نه توصیف قطعی آخرین نسخه

<a id="ref-33"></a>

۳۳. [r/LocalLLaMA — تجربهٔ استخراج اسناد عربی](https://www.reddit.com/r/LocalLLaMA/comments/1qiyxl4/we_tested_every_vlm_for_arabic_document/)؛ مشاهدهٔ خوداظهاری، بدون Benchmark مستقل قابل تعمیم به فارسی

<a id="ref-34"></a>

۳۴. [Qwen2.5-VL — معرفی رسمی](https://qwenlm.github.io/blog/qwen2.5-vl/)، [Qwen2.5-VL — Model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)؛ برای بررسی قرارداد مدل و توانایی‌های مستند

<a id="ref-35"></a>

۳۵. [NVIDIA Technical Blog — Approaches to PDF Data Extraction for Information Retrieval](https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval)؛ مقایسهٔ ارائه‌دهنده روی پیکربندی مشخص، نه Benchmark عمومی همهٔ VLMها

<a id="ref-36"></a>

۳۶. [PureDocBench — How Far Is Document Parsing from Solved?](https://arxiv.org/abs/2605.07492)؛ پیش‌چاپ ۲۰۲۶، شواهد پژوهشی و نه تست شخصی من

<a id="ref-37"></a>

۳۷. [r/LocalLLaMA — Docling PDF Parsing with remote VLM](https://www.reddit.com/r/LocalLLaMA/comments/1pkmi5v/docling_pdf_parsing_with_remote_vlm/)؛ تجربه و بحث دربارهٔ سازگاری و معیار کیفیت

<a id="ref-38"></a>

۳۸. [Persian Pixel — A large-scale synthetic OCR dataset for Persian language](https://arxiv.org/abs/2607.20385)، [Dataset card و شرایط دسترسی](https://huggingface.co/datasets/Omarrran/Persian_Pixel)؛ پیش‌چاپ پژوهشی ۲۰۲۶

<a id="ref-39"></a>

۳۹. [Thinking Machines Lab — Inkling 2 Mini model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)


<a id="ref-40"></a>

۴۰. [Python BiDi — documentation](https://python-bidi.readthedocs.io/en/latest/)، [Arabic Reshaper — repository and configuration](https://github.com/mpcabd/python-arabic-reshaper)، [Unicode normalization in Python](https://docs.python.org/3/library/unicodedata.html)

<a id="ref-41"></a>

۴۱. Docling issue reports: [#1938](https://github.com/docling-project/docling/issues/1938)، [#1421](https://github.com/docling-project/docling/issues/1421)، [#2179](https://github.com/docling-project/docling/issues/2179)، [#3021](https://github.com/docling-project/docling/issues/3021)، [#3462](https://github.com/docling-project/docling/issues/3462)؛ گزارش‌های نسخه و نمونه، نه معیار مستقل فارسی

<a id="ref-42"></a>

۴۲. [Bina family — published model catalog](https://huggingface.co/Reza2kn/models)، [Bina-0.4-Rizeh — public 193-crop reproduction and limitations](https://huggingface.co/Reza2kn/Bina-0.4-Rizeh)، [Bina-0.3-Koochik](https://huggingface.co/Reza2kn/Bina-0.3-Koochik)، [Bina-0.4-Koochik](https://huggingface.co/Reza2kn/Bina-0.4-Koochik)

<a id="ref-43"></a>

۴۳. [Google Document AI — processor list and version IDs](https://docs.cloud.google.com/document-ai/docs/processors-list)، [Release notes](https://docs.cloud.google.com/document-ai/docs/release-notes)، [Gemini API — media resolution](https://aistudio.google.com/docs/media-resolution)، [Gemini API — document understanding](https://ai.google.dev/gemini-api/docs/document-processing)

<a id="ref-44"></a>

۴۴. [ParsAiGroup — DeepSeek-OCR Persian fine-tune](https://huggingface.co/ParsAiGroup/deepseek_ocr_hezar4k_persian)؛ کارت مدل کامیونیتی؛ بدون ارزیابی مستقل در این مطلب

<a id="ref-45"></a>

۴۵. [NVIDIA Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)، [Nemotron Parse 2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0)، [Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse)؛ کارت‌های مدل و قرارداد خروجی

<a id="ref-46"></a>

۴۶. [LlamaParse — Parse overview](https://developers.llamaindex.ai/llamaparse/parse/)، [Response format](https://developers.llamaindex.ai/llamaparse/parse/guides/response-format/)


<a id="ref-47"></a>

۴۷. [Persian OCR comparison — Bina vs. Gemma vs. Qwen](https://github.com/alimohamad2005/persian-ocr-comparison)؛ مقایسهٔ اکتشافی با کنترل‌های مصنوعی و خروجی خام منتشرشده.

<a id="ref-48"></a>

۴۸. [Alireza Taheri — Evaluating OCR Engines on Handwritten Persian Documents](https://alireza-taherif.github.io/blog/evaluating-ocr-engines-on-handwritten-persian-documents/)؛ گزارش مقدماتی و اجرای ناقص روی دست‌خط.

<a id="ref-49"></a>

۴۹. [Gemini API — Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)، [Claude — Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)، [OpenAI — Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)؛ پشتیبانی Schema و محدودیت‌های API.

<a id="ref-50"></a>

۵۰. [AIIM — IDP Survey 2025](https://info.aiim.org/aiim-study-reveals-ai-driven-transformation-in-document-processing)، [Verdantix — Market Insight: Document AI Solutions For Enterprise](https://www.verdantix.com/client-portal/report/market-insight--document-ai-solutions-for-enterprise)؛ شواهد روند صنعت، نه سهم بازار VLM در استخراج.

<a id="ref-51"></a>

۵۱. [NVIDIA — Justt chargeback document extraction](https://developer.nvidia.com/case-studies/how-to-scale-chargeback-document-extraction-with-nvidia-nemotron-parse)، [Google Document AI — Gemini Layout Parser](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk)؛ نمونه‌های ترکیب مدل بصری، OCR و Pipeline.
