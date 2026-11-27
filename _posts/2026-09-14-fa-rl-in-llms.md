---
layout: article
lang: fa
title: "یادگیری تقویتی در LLMها؛ از RLHF تا Agentهای بلندمدت"
description: "تغییر مسیر RL در مدل‌های زبانی را از reward انسانی و verifier تا GRPO، Agentic RL، distillation، مدیریت trajectory و چالش‌های ارزیابی مرور می‌کنیم."
date: 2026-09-14
permalink: /fa/writing/rl-in-llms/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "محمد شجاعی"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# یادگیری تقویتی در LLMها؛ از RLHF تا Agentهای بلندمدت

یادگیری تقویتی در LLMها دیگر فقط مرحله‌ای برای تنظیم لحن دستیار نیست. حالا از آن برای تقویت reasoning، کدنویسی، جست‌وجو و کار چندمرحله‌ای Agentها استفاده می‌کنند. برای یک AI Engineer، سؤال اصلی فقط انتخاب PPO یا GRPO نیست؛ باید بدانیم مدل در چه محیطی تمرین می‌کند، نتیجه چطور بررسی می‌شود، چه بخشی از trajectory پاداش می‌گیرد و هزینهٔ اجرای این حلقه چقدر است.

## از ترجیح انسان تا نتیجهٔ قابل‌راستی‌آزمایی

در یک مسیر کلاسیک post-training، ابتدا مدل روی متن عمومی pretrain می‌شود، بعد با Supervised Fine-Tuning نمونهٔ پاسخ مطلوب می‌بیند و در ادامه با ترجیح انسانی یا RL تنظیم می‌شود. در RLHF، انسان چند پاسخ را مقایسه می‌کند، از آن ترجیح‌ها reward model می‌سازند و policy با الگوریتمی مثل PPO تلاش می‌کند reward بیشتری بگیرد.

این روش می‌تواند رفتار مدل را به ترجیح انسانی نزدیک کند، اما reward model خودِ حقیقت را نمی‌داند؛ فقط از دادهٔ ترجیح، یک برآورد یاد می‌گیرد. پاسخ قانع‌کننده ممکن است اشتباه باشد یا مدل یاد بگیرد ظاهر مطلوبی تولید کند. برای بعضی taskها می‌توان نتیجه را مستقیم بررسی کرد: کد اجرا و تست می‌شود، جواب ریاضی با checker سنجیده می‌شود یا تغییر state در محیط تأیید می‌شود. این ایده را معمولاً **RL with Verifiable Rewards (RLVR)** می‌نامند.

برای نمونه، DeepSeek-R1-Zero از نمونه‌های شناخته‌شدهٔ این مسیر است: مدل پایه با RL و rewardهای قابل‌بررسی برای ریاضی و کدنویسی تمرین شد، بدون این‌که پیش از آن مرحلهٔ SFT ویژهٔ reasoning داشته باشد. مقاله از GRPO استفاده می‌کند. در این روش، برای هر prompt چند پاسخ نمونه‌گیری می‌شود و هر پاسخ بر اساس reward خود نسبت به بقیهٔ گروه ارزیابی می‌شود؛ در قیاس با PPO کلاسیک، الگوریتم به یک value model جداگانه برای تخمین ارزش هر token وابسته نیست.

این حذف رایگان نیست. برای مقایسهٔ پاسخ‌ها چند rollout لازم است و برای trajectory بلند، تولید هر rollout می‌تواند زمان و token زیادی مصرف کند. ضمن این‌که RL لزوماً رفتار تازه‌ای را از هیچ خلق نمی‌کند: تحلیل‌های بعدی روی R1-Zero نشان می‌دهند بعضی نشانه‌های reasoning ممکن بود از قبل در Base Model وجود داشته باشند و RL احتمال بروز یا ادامهٔ آن‌ها را تغییر داده باشد.

## چرا GRPO را تغییر می‌دهند؟

هر الگوریتم یک failure mode را بهتر می‌کند و ممکن است trade-off دیگری بسازد. یک نقد پژوهشی به R1-Zero-like training نشان داد بعضی objectiveهای GRPO می‌توانند طول پاسخ را، حتی برای خروجی‌های نادرست، به‌شکل نامطلوب زیاد کنند. **Dr. GRPO** برای کم‌کردن همین bias در نرمال‌سازی و طول پاسخ پیشنهاد شد.

برای نمونه، **DAPO** یک recipe متن‌باز برای آموزش RL در مقیاس بالا ارائه می‌دهد که sampling و clipping را تغییر می‌دهد تا مسئله‌هایی مثل گروه‌های بی‌اطلاع، پاسخ‌های بیش‌ازحد بلند و پایداری آموزش را بهتر مدیریت کند. **GSPO** از زاویهٔ دیگری اهمیت ratio و clipping را در سطح sequence صورت‌بندی می‌کند. تیم Qwen گزارش کرد که این روش در آزمایش‌هایشان برای RL مدل‌های MoE پایدارتر و کارآمدتر بود؛ این گزارش تضمین نمی‌کند GSPO روی هر مدل یا داده بهتر باشد.

گزارش Qwen3 همچنین نشان می‌دهد post-training همیشه خطی نیست. در recipe آن‌ها SFT، RL، تولید و انتخاب پاسخ‌های مناسب و SFT دوباره در چند مرحله ترکیب می‌شوند. مدل می‌تواند رفتاری را در RL بیاموزد، نمونه‌های بهتر را برای SFT بعدی بسازد و سپس دوباره بهینه‌سازی شود. در نتیجه انتخاب الگوریتم باید همراه با نوع داده، budget، verifier و معیار ارزیابی انجام شود، نه فقط با نام روش.

## وقتی مدل باید با محیط تعامل کند

در RL برای Agent، پاسخ نهایی تنها چیزی نیست که اهمیت دارد. Agent ممکن است جست‌وجو کند، صفحه باز کند، ابزار فراخوانی کند، state را تغییر دهد و بر اساس نتیجه مسیر بعدی را انتخاب کند. این کار را می‌توان به‌شکل یک فرایند تصمیم‌گیری مرحله‌ای دید: مدل state را می‌بیند، action یا tool call می‌سازد، محیط نتیجه را برمی‌گرداند و مدل وارد مرحلهٔ بعد می‌شود.

گزارش Kimi-Researcher نمونه‌ای از آموزش end-to-end برای Agent پژوهشی را توصیف می‌کند که از search، browser و اجرای کد استفاده می‌کند و reward را بر اساس حل task می‌گیرد. مقالهٔ Agent-R1 هم اجزای RL برای Agentها را در قالب MDP صورت‌بندی و یک framework آزمایشی برای taskهای چندمرحله‌ای معرفی می‌کند. این نتایج نشان می‌دهند که policy، ابزار و environment به هم وابسته‌اند؛ اما نتیجهٔ یک benchmark پژوهشی به‌تنهایی کارایی هر Agent محصولی را ثابت نمی‌کند.

هرچه trajectory طولانی‌تر شود، credit assignment دشوارتر است: اگر صد action بعداً به شکست منجر شود، پاداش صفر نمی‌گوید کدام تصمیم مسیر را خراب کرد. برای reward صرفاً نهایی ممکن است signal خیلی کم‌تراکم باشد؛ reward مرحله‌ای یا verifier جزئی می‌تواند کمک کند، اما طراحی آن هم خطر تشویق رفتار اشتباه را دارد.

## هزینهٔ Agentic RL فقط GPU نیست

Agentهای بلندمدت به چند constraint عملیاتی می‌رسند: rolloutها طول یکسانی ندارند، بخشی از batch ممکن است منتظر کندترین اجرا بماند، context پر از خروجی ابزار می‌شود و reward دیر می‌رسد. مدیریت context، scheduling ناهمگام و checkpoint کردن trajectory بخشی از سیستم آموزش‌اند، نه جزئیات کم‌اهمیت اطراف optimizer.

در گزارش MiniMax، Forge نمونه‌ای از طراحی vendor برای این مسئله‌هاست. گزارش آن‌ها context management، reward مرتبط با completion time و اجرای asynchronous rolloutها را بخشی از زیرساخت آموزش Agent معرفی می‌کند. این شرح یک recipe صنعتی است؛ برای نتیجه‌گیری مستقل باید ablationها و اندازه‌گیری‌ها را در همان محیط مطالعه کرد.

روش‌های جدید هم trade-offهای متفاوتی پیشنهاد می‌کنند. **BASIS** برای reasoning از یک rollout به‌ازای هر prompt استفاده می‌کند و اطلاعات کل batch را برای برآورد advantage به‌کار می‌گیرد. **SAO** روی Agentic RL ناهمگام تمرکز دارد؛ در طراحی آن یک rollout با value model و ورود asynchronous داده‌ها ترکیب می‌شود. مقایسهٔ این روش‌ها باید هزینهٔ ساخت rollout، حافظهٔ critic، staleness داده و کیفیت task را کنار هم بسنجد؛ یک rollout همیشه از چند rollout ارزان‌تر یا بهتر نیست.

## از trajectory شکست‌خورده تا skill و distillation

یک مسیر تازه این است که تجربهٔ Agent را برای آموزش بعدی هم مصرف کنیم. **SEED** از trajectoryهای on-policy، skillهای زبانی می‌سازد؛ این skillها می‌توانند workflow، مشاهدهٔ تعیین‌کننده یا قاعدهٔ پرهیز از شکست را خلاصه کنند و سپس اثرشان از راه on-policy distillation به policy منتقل شود. در **SESA** یک challenger مسئله می‌سازد، solver آن را حل می‌کند و شکست‌های آموزنده به skill memory برمی‌گردند؛ در نتیجه حافظه و curriculum مسئله‌ها با هم تغییر می‌کنند.

این روش‌ها هنوز پژوهش‌اند و نتایجشان به taskها، مدل‌ها و پروتکل‌های خود مقاله محدود است. ذخیرهٔ skill هم مرز اعتماد تازه‌ای ایجاد می‌کند: skill اشتباه یا آلوده می‌تواند رفتارهای بعدی را جهت دهد، پس provenance، اعتبارسنجی و امکان حذف آن‌ها مهم است.

در مقیاس مدل‌های بزرگ، RL متخصص‌محور و distillation هم کنار هم قرار می‌گیرند. گزارش فنی Nemotron 3 Ultra از pipeline شامل SFT، RL و Multi-Teacher On-Policy Distillation (MOPD) می‌گوید؛ هدف، انتقال رفتار چند teacher متخصص به یک مدل عمومی روی trajectoryهای خود student است. در K2 Horizon نیز model card عمومی از پنج شاخهٔ RL تخصصی و ادغام آن‌ها گزارش می‌دهد، سپس چند مرحلهٔ SFT روی checkpoint ادغام‌شده را شرح می‌دهد.

## Verifier هم ممکن است اشتباه فهمیده شود

روش RLVR فقط وقتی مفید است که reward واقعاً هدف سیستم را اندازه بگیرد. اگر grader تنها ظاهر موفقیت را بسنجد، مدل ممکن است راهی برای پاس‌کردن آن پیدا کند بی‌آن‌که task را همان‌طور که کاربر می‌خواهد انجام دهد. تست ناقص، دادهٔ لو‌رفته، shortcut در محیط یا خروجی‌ای که فقط parser را راضی کند، همگی می‌توانند reward را منحرف کنند.

به همین دلیل benchmark را با چند نمونهٔ انسانی یا verifier ظاهراً دقیق تمام نکنید. taskهای adversarial بسازید، مسیر موفقیت را دستی نمونه‌برداری کنید، grader را با workaroundهای محتمل امتحان کنید و نتیجه را روی یک مجموعهٔ مستقل بسنجید. K2 Horizon model card در زمان آخرین بررسی این مقاله جزئیات یک ممیزی عددی از reward hacking ارائه نمی‌دهد؛ بنابراین در این‌جا عددی برای آن ادعا نمی‌کنیم.

## از کجا شروع کنیم؟

پیش از انتخاب optimizer، این موارد را تعریف کنید:

- **هدف قابل‌سنجش:** دقیقاً کدام state نهایی یا artifact باید درست باشد؟
- **Verifier:** چه کسی یا چه برنامه‌ای می‌تواند موفقیت را مستقل از ادعای مدل بررسی کند؟
- **هزینهٔ مسیر:** هر rollout چند token، فراخوانی ابزار، ثانیه و واحد compute مصرف می‌کند؟
- **سیگنال میانی:** آیا verifier مرحله‌ای قابل‌اعتماد دارید، یا reward فقط در پایان می‌رسد؟
- **محدودیت محیط:** سقف context، timeout، دسترسی ابزار، retry و حالت توقف چیست؟
- **ارزیابی:** چه مجموعهٔ مستقل و چه تحلیل failureای می‌تواند reward hacking یا افت توانایی را آشکار کند؟

یادگیری تقویتی برای LLM یک optimizer منفرد نیست؛ طراحی task، محیط، reward، verifier، context، rollout، evaluation و serving را به هم وصل می‌کند. برای یک AI Engineer مهم‌ترین سؤال این نیست که «کدام الگوریتم ترند شده؟»؛ باید پرسید «مدل چه رفتاری را در چه محیطی می‌آموزد و ما چطور می‌فهمیم آن رفتار واقعاً مفید است؟»

## منابع

- DeepSeek-AI، [DeepSeek-R1](https://arxiv.org/abs/2501.12948) و [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300).
- Liu و همکاران، [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783).
- Yu و همکاران، [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476).
- تیم Qwen، [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) و [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071).
- تیم Moonshot AI، [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/).
- Cheng و همکاران، [Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](https://arxiv.org/abs/2511.14460).
- MiniMax، [Forge: Scalable Agent RL Framework and Algorithm](https://www.minimax.io/blog/forge-scalable-agent-rl-en-1779896141).
- Dai و همکاران، [DR-Venus](https://arxiv.org/abs/2604.19859).
- Gong و همکاران، [BASIS](https://arxiv.org/abs/2605.27293).
- NVIDIA، [Nemotron 3 Ultra Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf).
- Hou و همکاران، [Single-Rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508).
- Wu و همکاران، [SEED: Self-Evolving On-Policy Distillation](https://arxiv.org/abs/2607.14777).
- Fu و همکاران، [Self-Play Meets Skill Evolution (SESA)](https://arxiv.org/abs/2607.29468).
- DeepSeek-AI، [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348).
- Institute of Foundation Models، [K2 Horizon](https://ifm.ai/blog/k2/) و [K2-Horizon-375B-A23B model card](https://huggingface.co/IFM/K2-Horizon-375B-A23B).
