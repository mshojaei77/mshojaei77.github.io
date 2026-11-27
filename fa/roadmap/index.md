---
layout: article
lang: fa
title: "نقشهٔ راه مهندسی هوش مصنوعی"
description: "مسیر عملی یادگیری LLM از مبانی تا استقرار."
permalink: /fa/roadmap/
nav_exclude: true
---
# نقشهٔ راه مهندسی هوش مصنوعی

این نقشهٔ راه برای یادگیری عملی توسعهٔ مدل زبانی، از شبکه‌های عصبی تا استقرار در Production است. با پیاده‌سازی و مثال‌های واقعی، مدل بسازید، آموزش دهید و مستقر کنید.

<img width="1037" height="418" alt="نمودار نسخهٔ اصلی نقشهٔ راه (انگلیسی)" src="https://github.com/user-attachments/assets/ceb1e3a3-06c7-46f1-87a3-304b8cab20a6" />

| بخش | موضوع‌های اصلی | زمان تخمینی |
|---|---|---|
| مبانی | شبکه‌های عصبی، Transformer، توکن‌سازی، Embedding، Attention | ۱۲ تا ۲۰ هفته |
| ساخت و آموزش مدل | آماده‌سازی داده، پیش‌آموزش، Fine-tuning، RLHF و هم‌ترازی ترجیحات | ۱۶ تا ۲۸ هفته |
| موضوعات پیشرفته و تخصص | ارزیابی، استدلال، کوانتیزه‌سازی، بهینه‌سازی Inference، معماری‌ها | ۲۰ تا ۳۶ هفته |
| مهندسی و کاربردها | استقرار، RAG، ایجنت‌ها، چندوجهی، امنیت، LLMOps | ۱۲ تا ۲۴ هفته |

**زمان کلی تخمینی:** ۱۲ تا ۲۴ ماه، بسته به تجربهٔ قبلی و زمانی که اختصاص می‌دهید.

**مسیر انعطاف‌پذیر:** بخش‌ها مستقل‌اند؛ با توجه به پیشینه و هدف خود شروع کنید:
- **کاملاً تازه‌کار:** بخش اول برای مبانی کامل.
- **با تجربهٔ ML/AI:** بخش دوم یا موضوع موردعلاقه.
- **تمرکز مهندسی:** بخش چهارم برای کاربردهای عملی.
- **علاقه به پژوهش:** بخش سوم برای موضوعات پیشرفته.

### پیش‌نیازها

**برنامه‌نویسی و توسعه**
- **Python:** [آموزش مبانی](https://www.python.org/about/gettingstarted/) و [Automate the Boring Stuff](https://automatetheboringstuff.com/)
- **Git و کنترل نسخه:** [آموزش GitHub](https://try.github.io/) و [Pro Git](https://git-scm.com/book)
- **خط فرمان:** [The Missing Semester](https://missing.csail.mit.edu/) و [آموزش فشردهٔ خط فرمان](https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Understanding_client-side_tools/Command_line)
- **SQL و پایگاه داده:** [آموزش تعاملی SQLBolt](https://sqlbolt.com/) (اختیاری)

**ریاضی**
- **جبر خطی:** [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **آمار پایه:** [StatQuest](https://www.youtube.com/@statquest) و [Khan Academy](https://www.khanacademy.org/math/statistics-probability)
- **حسابان:** [حسابان تک‌متغیرهٔ MIT](https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/) (اختیاری)

**یادگیری ماشین**
- **مبانی ML:** [CS229 Stanford](https://cs229.stanford.edu/) و [دورهٔ Andrew Ng](https://www.coursera.org/learn/machine-learning) (اختیاری)
- **یادگیری عمیق:** [Dive into Deep Learning](https://d2l.ai/) و [fast.ai](https://course.fast.ai/) (اختیاری)

---

# بخش اول: مبانی

**مطالب مرتبط:** [معماری LLM از Attention تا MoE](/fa/writing/llm-architecture/)، [مقایسهٔ Embeddingهای فارسی](/fa/writing/persian-embedding-models/) و [یادداشت مهندسی دربارهٔ انتخاب مدل](/fa/writing/ai-engineering-week-01-preface-reading-notes/).

<img width="1026" height="443" alt="نمودار نسخهٔ اصلی نقشهٔ راه (انگلیسی)" src="https://github.com/user-attachments/assets/3abad5b8-23b8-4388-baf1-c5e28efec422" />

**تمرکز:** مفاهیم اصلی ML، شبکه‌های عصبی، مدل‌های سنتی، توکن‌سازی، Embedding و Transformer.  
**سطح:** مقدماتی تا متوسط.  
**خروجی مورد انتظار:** پایهٔ محکم ML/NLP و معماری Transformer.

**هدف یادگیری:** با پیاده‌سازی عملی، از شبکه‌های عصبی شروع کنید، سیر تحول مدل‌های زبانی سنتی تا Transformer را بفهمید و بر توکن‌سازی، Embedding و معماری Transformer مسلط شوید.

## ۱. مبانی شبکه‌های عصبی برای LLM
**سطح:** مقدماتی تا متوسط | **پیش‌نیاز:** حسابان پایه، جبر خطی، Python

### اجزای اصلی
- نورون و پرسپترون: واحد پایه (جمع وزن‌دار و Bias).
- لایه‌ها و شبکه‌های Feed-forward: ورودی، پنهان، خروجی؛ MLP به‌عنوان محاسبهٔ اصلی بلوک Transformer.
- تابع فعال‌سازی: ReLU (اصلی)، GELU/Swish (در LLMهای جدید مثل GPT/BERT برای گرادیان بهتر)، Sigmoid/Tanh (تاریخی).

### مبانی آموزش
- Forward Pass و پیش‌بینی.
- تابع Loss: Cross-entropy (ضروری برای مدل‌سازی زبان و پیش‌بینی توکن بعدی).
- Backpropagation: قاعدهٔ زنجیره‌ای برای محاسبهٔ گرادیان.
- انواع Gradient Descent: SGD، Adam/AdamW (بهینه‌سازهای رایج آموزش LLM).

### بهینه‌سازی و پایداری
- گرادیان محوشونده/انفجاری: علت و روش‌های کاهش.
- مقداردهی اولیهٔ وزن: Xavier/He (مهم برای شبکهٔ عمیق).
- Regularization: Dropout (رایج در Transformer)، L2 Weight Decay.
- نرمال‌سازی: Layer Normalization (جزء اصلی Transformer)، Batch Norm (برای شناخت زمینه).

### اجزای معماری مرتبط با LLM
- اتصال Residual: اتصال میان‌بُر برای آموزش شبکهٔ عمیق‌تر.
- آموزش Mixed Precision: FP16/BF16 برای بازده مدل بزرگ.
- مبانی تنظیم هایپرپارامتر: اثر Learning Rate و Batch Size.

## ۲. مدل‌های زبانی سنتی
**سطح:** متوسط | **پیش‌نیاز:** احتمال و آمار

### مدل‌های N-gram
مدل‌سازی آماری زبان، Smoothing با Laplace، Good-Turing و Kneser-Ney، ارزیابی Perplexity.

### شبکه‌های Feedforward
مدل‌های عصبی اولیهٔ زبان، Context Window ثابت و نمایش توزیع‌شده.

### معماری‌های بازگشتی
RNN، LSTM، GRU، RNN دوسویه و چندلایه، چالش‌های مدل‌سازی دنباله.

### مدل‌های Sequence-to-sequence
Encoder-decoder، ترجمهٔ ماشینی، خلاصه‌سازی، Beam Search.

### Attention
نوآوری کلیدی منتهی به Transformer، Alignment و بردارهای زمینه.

### کاربردها
تولید متن، تحلیل احساسات، مدل‌سازی در سطح کاراکتر یا واژه، پیش‌بینی سری زمانی.

## ۳. توکن‌سازی
**سطح:** مقدماتی | **پیش‌نیاز:** مبانی Python

### مبانی
توکن، رویکرد کاراکتر/واژه/Subword، مدیریت واژگان و بهینه‌سازی Context Window.

### پیش‌پردازش
نرمال‌سازی، Pre-tokenization، پیش‌پردازش متن، مدیریت کاراکتر ویژه، ایموجی و قطعهٔ کد.

### الگوریتم‌های Subword
BPE، WordPiece، Unigram، SentencePiece و BPE در سطح Byte.

### روش‌های پیشرفته
توکن‌سازی چندزبانه، توکن‌سازی بصری، انتقال Tokenizer (TokenAdapt)، سازگارسازی دامنه.

### پیاده‌سازی
Encoding و Decoding، بهینه‌سازی اندازهٔ واژگان، بازده توکن‌سازی، واژگان خارج از فرهنگ (OOV).

### ابزارهای جدید
Hugging Face Tokenizers، tiktoken، توکن‌سازی GPT-4/Llama/BERT، پردازش اصطلاحات حقوقی و پزشکی.

## ۴. Embedding
**سطح:** مقدماتی تا متوسط | **پیش‌نیاز:** جبر خطی، Python

### Embedding ایستا
معماری Word2Vec، GloVe، FastText و نمایش‌های برداری سنتی.

### Embedding وابسته به زمینه
BERT، RoBERTa، Embedding مبتنی بر Transformer و نمایش وابسته به Context.

### Embedding چندوجهی
CLIP، ALIGN، جست‌وجوی بین‌وجهی، Embedding مشترک متن و تصویر، هم‌ترازی بینایی و زبان.

### عملیات برداری
قیاس معنایی، حساب برداری، معیار شباهت Cosine، Euclidean و Dot Product.

### کاربردها
جست‌وجوی معنایی، Dense و Sparse Retrieval، شباهت متن، سیستم پیشنهاددهنده.

### سازگارسازی دامنه
Fine-tuning Embedding، تحلیل احساسات مالی، جست‌وجوی فروشگاهی، مقاله‌های arXiv و دامنه‌های تخصصی.

## ۵. معماری Transformer
**سطح:** پیشرفته | **پیش‌نیاز:** شبکه‌های عصبی، جبر خطی

### مبانی معماری
پشتهٔ Encoder-decoder، گونه‌های Decoder-only، Layer Normalization و اتصال Residual.

### Attention
Self-attention، Multi-head، Scaled Dot-product، Cross-attention و Causal Attention.

### اجزای اصلی
ماتریس Query/Key/Value، الگوها و وزن‌های Attention، مصورسازی Attention.

### Positional Encoding
Sinusoidal، کدگذاری یادگرفتنی، RoPE و ALiBi.

### Attention پیشرفته
FlashAttention، MQA، GQA و KV Cache.

### تولید متن
Nucleus Sampling، Top-k، Beam Search، Masked Attention و پیاده‌سازی mini-GPT.

---

# بخش دوم: ساخت و آموزش مدل

**مطالب مرتبط:** [پس‌آموزش LLM از SFT تا یادگیری تقویتی](/fa/writing/llm-post-training/)، [Fine-tuning کم‌هزینه با LoRA، QLoRA و DoRA](/fa/writing/efficient-fine-tuning/) و [تقطیر دانش در LLM](/fa/writing/knowledge-distillation/).

<img width="1020" height="450" alt="نمودار نسخهٔ اصلی نقشهٔ راه (انگلیسی)" src="https://github.com/user-attachments/assets/f2fc0e54-d150-4610-ba52-b0e8bd4ea4f5" />

**تمرکز:** آماده‌سازی داده، پیش‌آموزش، Fine-tuning و هم‌ترازی ترجیحات.  
**سطح:** متوسط تا پیشرفته.  
**خروجی مورد انتظار:** توان آموزش و تنظیم دقیق مدل زبانی از صفر.

**هدف یادگیری:** دیتاست باکیفیت بسازید، پیش‌آموزش توزیع‌شده پیاده کنید، دیتاست دستور ایجاد کنید، SFT انجام دهید و با RLHF و DPO مدل را با ترجیحات انسان هم‌تراز کنید.

## ۶. آماده‌سازی داده
**سطح:** متوسط | **پیش‌نیاز:** Python، SQL

### جمع‌آوری داده
Web Scraping، API، تجمیع داده، Beautiful Soup، Scrapy، Common Crawl و C4.

### پردازش داده
پاک‌سازی، فیلتر، سنجش کیفیت، تشخیص آلودگی داده، مدیریت خطا و Rate Limit.

### حذف تکرار
MinHash، LSH، ایندکس افزایشی و بهینه‌سازی ساختار داده.

### کنترل کیفیت
اعتبارسنجی، امتیاز کیفیت، TinyStories، دادهٔ املاک و دستورالعمل Annotation.

### حریم خصوصی
تشخیص PII، NER و مدیریت اطلاعات هویتی.

### روش‌های پیشرفته
دادهٔ مصنوعی، Data Augmentation، سنجش کیفیت با ML و Pipeline داده.

## ۷. پیش‌آموزش LLM
**سطح:** تخصصی | **پیش‌نیاز:** Transformer، سیستم‌های توزیع‌شده

### اهداف آموزش
CLM، MLM، Prefix LM و پیش‌آموزش بدون نظارت.

### آموزش توزیع‌شده
Data Parallelism، Model Parallelism، Pipeline Parallelism، آموزش چندگره‌ای، Slurm، Kubernetes.

### بهینه‌سازی
ZeRO، Gradient Checkpointing، Mixed Precision، DeepSpeed، FairScale، Megatron-LM و Colossal-AI.

### مقیاس و بازده
Scaling Laws، Curriculum Learning، زمان‌بندی داده، بهینه‌سازی محاسبات و Throughput آموزش.

### زیرساخت
مدیریت Checkpoint، پایش Loss، ناپایداری آموزش، مکانیزم بازیابی و پیکربندی سخت‌افزار.

### روش‌های پیشرفته
آموزش استدلال ریاضی، بهینه‌سازی سرعت همگرایی و مصرف حافظه.

## ۸. دیتاست پس‌آموزش (برای Fine-tuning)
**سطح:** متوسط | **پیش‌نیاز:** آماده‌سازی داده

### فرمت دیتاست
دیتاست Instruction، Alpaca، ShareGPT، Chat Template و قالب مکالمه.

### ساخت داده
دادهٔ مصنوعی، کنترل کیفیت، فیلتر، Annotation و سیستم اعتبارسنجی.

### طراحی مکالمه
مکالمهٔ چندنوبتی، System Prompt، پیام کاربر/ربات، توکن ویژه و مدل Role-playing.

### تضمین کیفیت
امتیاز کیفیت، ترکیب دیتاست، بهینه‌سازی برای قابلیت خاص و ارزیابی کیفیت پاسخ.

### قابلیت‌های پیشرفته
افکار داخلی، جریان مکالمه، الگوهای تخصصی، مدیریت زمینه و تاریخچهٔ مکالمه.

### گزینش داده
راهنمای Annotation، سلامت دیتاست، تشخیص سوگیری، Curation و Benchmark کیفیت.

## ۹. Supervised Fine-tuning (SFT)
**سطح:** پیشرفته | **پیش‌نیاز:** مبانی پیش‌آموزش

### رویکرد Fine-tuning
تنظیم دقیق کامل در برابر PEFT و بهینه‌سازی منابع.

### PEFT پیشرفته
LoRA، QLoRA، Adapter، کوانتیزه‌سازی ۴ بیتی و محدودیت GPU مصرفی.

### آموزش تخصصی
Instruction Tuning، آموزش مدل چت، سازگارسازی دامنه، یادگیری مداوم و CodeLlama.

### ترکیب مدل
Model Merging، SLERP، TIES-Merging، DARE، مدل چندوظیفه‌ای و یکپارچه‌سازی مدل تخصصی.

### بهینه‌سازی حافظه
Gradient Checkpointing، FSDP و آموزش کم‌مصرف از نظر حافظه.

### کاربردها
تولید کد، مدل وظیفه‌محور، جلوگیری از فراموشی فاجعه‌بار و حفظ عملکرد.

## ۱۰. هم‌ترازی ترجیحات (RL Fine-tuning)
**سطح:** تخصصی | **پیش‌نیاز:** مبانی یادگیری تقویتی

### مبانی RL
مبانی و RL عمیق، بهینه‌سازی Policy و PPO.

### Pipeline RLHF
یادگیری تقویتی از بازخورد انسان، آموزش Reward Model، ترجیحات انسانی و فرایند سه‌مرحله‌ای.

### روش‌های جایگزین
DPO، Rejection Sampling و KTO (Kahneman-Tversky Optimization).

### ایمنی و هم‌ترازی
مفید بودن، بی‌ضرر بودن، صداقت، Constitutional AI، بازخورد AI و هم‌ترازی مبتنی بر اصول.

### ارزیابی
ارزیابی ایمنی و هم‌ترازی، دیتاست ترجیحات، Win Rate، Benchmark ایمنی و Alignment Tax.

### روش‌های پیشرفته
ارزیابی خودکار و انسانی، Self-critique، بازبینی پاسخ و اصول تعریف‌شده.

---

# بخش سوم: موضوعات پیشرفته و تخصص

**مطالب مرتبط:** [ارزیابی توانایی مدل‌های زبانی در فارسی](/fa/writing/persian-llm-evaluation/)، [ارزیابی استدلال مدل‌ها](/fa/writing/ai-reasoning-evaluation/) و [اجرای محلی LLM](/fa/writing/local-llm-inference/).

<img width="1019" height="419" alt="نمودار نسخهٔ اصلی نقشهٔ راه (انگلیسی)" src="https://github.com/user-attachments/assets/866d3789-96ed-4ae6-8140-8edb0b828048" />

**تمرکز:** ارزیابی، استدلال، بهینه‌سازی، معماری و بهبود مدل.  
**سطح:** تخصصی/پژوهشی.  
**خروجی مورد انتظار:** توان پژوهشی، انتشار مقاله و هدایت پیشرفت‌های نظری.

**هدف یادگیری:** توسعهٔ تخصص پژوهشی در ارزیابی LLM، تقویت استدلال، بهینه‌سازی مدل، معماری‌های جدید و روش‌های بهبود مدل برای تحقیق و توسعهٔ پیشرو.

## ۱۱. ارزیابی مدل
**سطح:** متوسط | **پیش‌نیاز:** آمار، آموزش مدل

### مجموعه Benchmark
MMLU، GSM8K، HumanEval، BigBench، Benchmark استاندارد و پروتکل ارزیابی دانشگاهی.

### ارزیابی انسانی
Crowdsourcing، قضاوت انسان، Chatbot Arena، مقایسهٔ رودررو و دستورالعمل Annotation.

### ارزیابی خودکار
LLM-as-judge، مقایسهٔ مدل، سنجش خودکار، معیار کیفیت و تحلیل آماری.

### ایمنی و انصاف
تست سوگیری و ایمنی، تشخیص محتوای سمی، BOLD، RealToxicityPrompts و چارچوب‌های انصاف.

### معیار عملکرد
Accuracy، F1، BLEU، ROUGE، Win Rate، ارزیابی مقایسه‌ای و سنجش کیفیت.

### زیرساخت ارزیابی
Framework ارزیابی، Benchmark سفارشی، ارزیابی دامنه‌محور و اعتبارسنجی.

## ۱۲. استدلال
**سطح:** متوسط | **پیش‌نیاز:** مهندسی پرامپت

### استدلال پیشرفته
System 2، Chain-of-Thought، Tree-of-thoughts و پرامپت‌نویسی پیشرفته.

### RL برای استدلال
Process Reward Model (PRM)، پاداش سطح گام، STEP-RLHF و GRPO.

### خودبهبود
Self-reflection، حلقهٔ Self-consistency، بودجهٔ deliberation و Test-time Compute Scaling.

### الگوهای معماری
Planner-worker، جداسازی Plan-work-solve و Monte Carlo Tree Search (MCTS).

### رویکردهای آموزش
دادهٔ مصنوعی استدلال، Self-training با Bootstrapping، Curriculum و آموزش چندمرحله‌ای.

### کاربردها
استدلال ریاضی و منطقی، تولید کد، سؤال Multi-hop و تجزیهٔ مسئله.

## ۱۳. کوانتیزه‌سازی
**سطح:** متوسط | **پیش‌نیاز:** بهینه‌سازی مدل

### نظریه
کاهش Precision، مبانی کوانتیزه‌سازی، Trade-off اندازه و دقت، Calibration.

### روش‌ها
PTQ، QAT، کوانتیزه‌سازی عدد صحیح و Sparsity.

### الگوریتم‌های پیشرفته
GPTQ، AWQ، SmoothQuant، ZeroQuant، INT4/INT8 و بهینه‌سازی مخصوص سخت‌افزار.

### فرمت استقرار
GGUF، llama.cpp، استقرار موبایل و Edge، پشتیبانی GPU مصرفی.

### بهینه‌سازی عملکرد
بهینه‌سازی حافظه، تسریع Inference، فشرده‌سازی مدل و محیط محدود از نظر منابع.

### سنجش کیفیت
ارزیابی کیفیت کوانتیزه‌سازی، Trade-off عملکرد، یکپارچه‌سازی ONNX و TensorRT.

## ۱۴. بهینه‌سازی Inference
**سطح:** پیشرفته | **پیش‌نیاز:** استقرار مدل

### بهینه‌سازی حافظه
FlashAttention، پیاده‌سازی و مدیریت KV Cache و مدیریت پیشرفتهٔ حافظه.

### تکنیک‌های عملکرد
Speculative Decoding، نمونه‌گیری موازی، Draft Model، Verifier و بهینه‌سازی سرعت.

### Batching
Dynamic و Continuous Batching، Inference چند GPU و چندگره‌ای، Load Balancing.

### Serving در Production
vLLM، TensorRT-LLM، DeepSpeed-Inference، Triton و سرورهای Throughput بالا.

### روش‌های پیشرفته
Test-time Preference Optimization (TPO)، فشرده‌سازی، تخصیص منابع و مقیاس‌دهی.

### بهینه‌سازی سخت‌افزار
CUDA، بهینه‌سازی تأخیر، Attention کم‌مصرف و پردازش دنبالهٔ بلند.

## ۱۵. انواع معماری مدل
**سطح:** پیشرفته | **پیش‌نیاز:** معماری Transformer

### معماری Sparse
MoE، شبکهٔ Expert، شبکهٔ Gating، Load Balancing و Trade-off Sparse/Dense.

### مدل فضای حالت
Mamba، RWKV، Selective State Space و پردازش دنباله در زمان خطی.

### زمینهٔ طولانی
Sliding Window Attention، Longformer، BigBird، پردازش سند بلند و Interpolation/Extrapolation.

### معماری ترکیبی
GraphFormers، LLM گراف‌محور، معماری جدید و ترکیب Convolution و Attention.

### نوآوری کارایی
بهینه‌سازی حافظه و محاسبه، Framework جست‌وجوی معماری و Benchmark عملکرد.

### گونه‌های پیشرفته
اجزای ترکیبی، گونه‌های معماری، جست‌وجوی معماری کارآمد و واحد پردازش تخصصی.

## ۱۶. بهبود مدل
**سطح:** پیشرفته | **پیش‌نیاز:** آموزش و بهینه‌سازی مدل

### گسترش زمینه
YaRN، Position Interpolation، دادهٔ متن بلند و یادآوری اطلاعات.

### ترکیب مدل
Model Merging، Ensembling، TIES-Merging، DARE، بازیابی افت مدل و سیستم یکپارچه.

### انتقال دانش
Knowledge Distillation، آموزش Teacher-student، فشرده‌سازی و بهینه‌سازی استقرار موبایل.

### یادگیری مداوم
سازگارسازی، Continual Learning، جلوگیری از فراموشی فاجعه‌بار و Meta-learning.

### خودبهبود
Self-improvement، بازخورد کاربر، تعامل، بهبود مستمر و ارتقای عملکرد.

### روش‌های پیشرفته
Ensemble، یکپارچه‌سازی مدل تخصصی، استقرار مدل فشرده و استراتژی بازیابی.

---

# بخش چهارم: مهندسی و کاربردها

**مطالب مرتبط:** [معماری Production RAG](/fa/writing/production-rag-architecture/)، [پردازش اسناد فارسی](/fa/writing/document-parsing/)، [Agent یا Workflow؟](/fa/writing/agent-vs-workflow/) و [شکست Agentها در محیط عملیاتی](/fa/writing/agent-production-failures/).

<img width="1024" height="446" alt="نمودار نسخهٔ اصلی نقشهٔ راه (انگلیسی)" src="https://github.com/user-attachments/assets/1d7ee1a8-3981-478f-9ba6-1a5143ff98ec" />

**تمرکز:** استقرار Production، RAG، ایجنت، چندوجهی، امنیت و عملیات.  
**سطح:** متوسط تا پیشرفته.  
**خروجی مورد انتظار:** اپلیکیشن‌ها و سیستم‌های LLM آمادهٔ Production در مقیاس.

**هدف یادگیری:** بهینه‌سازی استقرار و تسریع Inference، ساخت اپلیکیشن با RAG و ایجنت، یکپارچه‌سازی چندوجهی، LLMOps و AI مسئولانه برای راهکارهای مقیاس‌پذیر.

## ۱۷. اجرای LLM و ساخت اپلیکیشن
**سطح:** متوسط | **پیش‌نیاز:** توسعهٔ وب، API

### یکپارچه‌سازی API
API مدل زبانی، OpenAI/Anthropic، Rate Limit، احراز هویت، هزینه و کاربران هم‌زمان.

### توسعهٔ اپلیکیشن
چت‌بات حافظه‌دار، دمو تعاملی، سرور Production، Serving و REST API.

### مهندسی پرامپت
خروجی ساختاریافته، JSON Mode، Function Calling، مدیریت پرامپت و رابط مکالمه.

### استقرار
اجرای محلی، مدل متن‌باز، Streaming، اپلیکیشن بلادرنگ و Test-time Autoscaling.

### Frameworkها
FastAPI، Flask، Streamlit، Gradio، WebSocket، معماری اپلیکیشن و بهینه‌سازی مقیاس‌پذیری.

### زیرساخت
Containerization، Docker، Kubernetes، Load Balancing، مدیریت منابع، پایش و پشتیبانی چندوجهی.

## ۱۸. RAG (تولید با کمک بازیابی)
**سطح:** پیشرفته | **پیش‌نیاز:** Embedding، پایگاه داده

### پردازش سند
ورود داده، منابع داده، Chunking، پردازش PDF، صفحهٔ وب و پایگاه داده.

### عملیات برداری
مدل Embedding، نمایش برداری، Vector DB (Pinecone، Weaviate، Chroma) و راهکار ذخیره‌سازی.

### روش بازیابی
پیاده‌سازی Retrieval، BM25، جست‌وجوی معنایی و ترکیبی، Reranking و بهبود Query.

### RAG پیشرفته
Graph RAG، Knowledge Graph، ساخت گراف دانش و Intelligent Document Processing (IDP).

### RAG مکالمه‌ای
مکالمهٔ چندنوبتی، Agentic RAG، برنامه‌ریزی Query، استدلال چندگامی و مدیریت زمینه.

### زیرساخت
مقیاس‌دهی، Cache، پایش، ایندکس افزایشی، به‌روزرسانی اسناد، معیار ارزیابی و دقت بازیابی.

## ۱۹. استفاده از ابزار و ایجنت AI
**سطح:** پیشرفته | **پیش‌نیاز:** Function Calling، برنامه‌ریزی

### Function Calling
استفاده از ابزار، API فراخوانی تابع، یکپارچه‌سازی ابزار خارجی و دسترسی بلادرنگ به اطلاعات.

### معماری ایجنت
پیاده‌سازی ایجنت، سیستم برنامه‌ریزی، استدلال، اجرای خودمختار وظیفه، ایمنی و کنترل.

### Frameworkها
ReAct، LangGraph، AutoGen، CrewAI، برنامه‌ریزی، استفاده از ابزار و هماهنگی چندایجنتی.

### ایجنت تخصصی
تحلیل بازار مالی، دستیار برنامه‌نویسی، دستیار پژوهش، دستگاه خانهٔ هوشمند و اتوماسیون Workflow.

### قابلیت‌های پیشرفته
ارتباط بین ایجنت‌ها، هماهنگی وظیفه، مدیریت خطا، اصلاح تکرارشونده و حل چندمرحله‌ای مسئله.

### ملاحظات Production
محدودیت ایمنی، نظارت انسانی، گردش‌کار تأیید، اعتبارسنجی و مدیریت فرمان مبهم.

## ۲۰. LLM چندوجهی
**سطح:** پیشرفته | **پیش‌نیاز:** بینایی ماشین، پردازش صوت

### مدل بینایی-زبان
CLIP، LLaVA، GPT-4V، Transformer چندوجهی و نمایش مشترک تصویر و متن.

### ورودی چندوجهی
ورودی متن، صوت و تصویر، Transfer Learning، مدل پیش‌آموزش‌دیده و Feature Fusion.

### پردازش تصویر
Image Captioning، VQA، استدلال بصری، درک سند و OCR.

### یکپارچه‌سازی صوت
پردازش صوت، گفتار، torchaudio، Whisper و Cross-modal Attention.

### تولید
تولید تصویر از متن، Stable Diffusion، مهندسی پرامپت، Negative Prompt و تنظیم پارامتر.

### کاربردها
چت‌بات و ایجنت چندوجهی، تحلیل سند، تحلیل اسکرین‌شات کد و چت‌بات فروشگاهی.

## ۲۱. امنیت LLM و AI مسئولانه (اختیاری)
**سطح:** پیشرفته | **پیش‌نیاز:** مبانی امنیت، AI اخلاقی

### چارچوب امنیت
OWASP LLM Top 10، مسیر حمله، ارزیابی آسیب‌پذیری، کنترل امنیت و لایهٔ دفاعی.

### روش‌های حمله
Prompt Injection، Jailbreak، نشت داده، مسموم‌سازی دادهٔ آموزشی و Backdoor.

### استراتژی دفاع
دفاع در برابر Prompt Injection، کاهش Jailbreak، پاک‌سازی ورودی، فیلتر خروجی و Guardrail.

### حریم خصوصی
پوشاندن اطلاعات شخصی، روش‌های بازسازی، Differential Privacy و Federated Learning.

### AI مسئولانه
انصاف، تشخیص و کاهش سوگیری، حاکمیت AI و انطباق (GDPR، CCPA).

### تست امنیت
Frameworkهای Red Teaming، شبیه‌سازی حمله، اعتبارسنجی امنیت، تعدیل محتوا و کنترل ایمنی.

## ۲۲. LLMOps
**سطح:** پیشرفته | **پیش‌نیاز:** DevOps، MLOps و پلتفرم‌های ابری

### مدیریت مدل
نسخه‌بندی مدل، Model Registry (MLflow)، یکپارچه‌سازی Hugging Face Hub، ساخت Model Card و کنترل نسخه.

### CI/CD
تست خودکار، GitHub Actions، فرایند استقرار، Workflow خودکار و مدیریت وابستگی.

### زیرساخت
Docker، OpenShift، Containerization، Kubernetes، Terraform، Apache Spark و بهینه‌سازی Inference.

### پایش
ابزار مشاهده‌پذیری LLM، Prometheus، Grafana، معیار سفارشی، داشبورد بلادرنگ و هشدار.

### آزمایش
Frameworkهای A/B Test، پلتفرم آزمایش، تحلیل آماری، مقایسهٔ مدل و بهینه‌سازی پرامپت.

### بهینه‌سازی هزینه
ثبت هزینه، تخصیص منابع، مقیاس‌دهی خودکار و مبتنی بر تقاضا، تحلیل معیار عملکرد.

---

**مشارکت:**
- **بهبود محتوا:** پیشنهادها را با Issue یا PR در GitHub ارسال کنید.
- **گفت‌وگو:** [به گروه یادگیری و بحث‌های فنی بپیوندید](https://t.me/AI_LLMs).
- **اشتراک‌گذاری:** کمک کنید دیگران هم نقشهٔ راه را پیدا کنند.
- **بازخورد:** تجربهٔ یادگیری شما به بهبود محتوا کمک می‌کند.

**تشکر:** از جامعهٔ متن‌باز، پژوهشگران و مهندسانی که توسعهٔ LLM را برای همه دسترس‌پذیر می‌کنند.
