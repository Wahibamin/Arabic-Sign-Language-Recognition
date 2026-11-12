# دليل النشر — GitHub والواجهة العامة

## أولاً: تشغيل المشروع محلياً

**الحد الأدنى:** Python 3.11، وبيئة يمكنها تثبيت TensorFlow. يفضّل توفر ذاكرة كافية لتحميل شبكتَي CNN معاً. التشغيل على CPU ممكن وقد يكون أبطأ من GPU.

```powershell
cd Arabic_Sign_Language_V5_GitHub_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify_assets.py --strict
python app.py
```

افتح `http://127.0.0.1:7860`، ثم ارفع صورة أو استخدم الكاميرا. **لا تعاد عملية التدريب** عند تشغيل هذه الأوامر.

## ثانياً: رفع المشروع إلى حساب GitHub

1. أنشئ مستودعاً جديداً فارغاً على https://github.com/new، مثل `arabic-sign-language-v5`.
2. فك ضغط الحزمة الكاملة على جهازك.
3. افتح PowerShell داخل مجلد المشروع.
4. نفذ `powershell -ExecutionPolicy Bypass -File .\scripts\publish_github.ps1` بعد مراجعة محتواه، أو استخدم الأوامر اليدوية الآتية:

```bash
git init -b main
git add .
git commit -m "Publish ArSL V5 final project"
git remote add origin https://github.com/USERNAME/arabic-sign-language-v5.git
git push -u origin main
```

استبدل `USERNAME` باسم حسابك والمستودع بعنوانك الفعلي. يتطلب الدفع تسجيل دخول GitHub أو اعتماداً مناسباً؛ لا تُكتب كلمة مرور أو رمز دخول داخل الكود.

**لماذا ليس الرفع بالمتصفح؟** تحدد GitHub حجم الملف المرفوع مباشرة عبر المتصفح بـ **25 MiB**، أما Git CLI فيقبل حتى **100 MiB لكل ملف**. يوجد في الحزمة `model_1.keras` أكبر من 25 MiB وأقل من 100 MiB، لذلك يناسب Git CLI. للأعمال المتكررة وإصدارات النماذج الكبيرة يمكن استخدام Git LFS بعد تثبيته.

مرجع رسمي: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

## ثالثاً: نشر موقع تفاعلي عام باستخدام Hugging Face Spaces

**مهم:** GitHub Pages لا يشغّل تطبيق Python/Gradio وTensorFlow. لعرض النموذج في رابط ويب عام استعمل Hugging Face Spaces (أو خادماً آخر يدعم Python).

1. افتح https://huggingface.co/new-space وأنشئ Space جديداً.
2. اختر `Gradio` كتقنية SDK.
3. ارفع **نفس ملفات المستودع** (يشمل `app.py`، `requirements.txt`، `src/`، `models/`).
4. يحتوي `README.md` في المستودع على YAML خاص بـ Spaces: `sdk: gradio` و`app_file: app.py` و`python_version: 3.11`.
5. تابع سجلات البناء والتشغيل. قد يستغرق تثبيت TensorFlow وتحميل الشبكتين وقتاً، وقد تحتاج استضافة بموارد أكبر إذا لم تكفِ موارد الخطة المجانية.
6. اختبر رفع صورة والتقاط صورة من الهاتف. الكاميرا تتطلب أذونات المتصفح وقد لا تعمل عبر HTTP غير آمن.

المصدر: https://huggingface.co/docs/hub/spaces-sdks-gradio

## خصوصية ورخصة النشر

- اعرض تنبيهاً بأن الصور تُعالج على الخادم المستضيف.
- تحقق من حقوق إعادة توزيع الأوزان وقاعدة AASL قبل نشر مستودع عام.
- لا تضف البيانات الخام أو ملفات شخصية/مفاتيح/رموز دخول إلى GitHub.
- ملفات `joblib` لا تُحمّل إلا من مصدر موثوق.

## حلول سريعة للمشكلات

| المشكلة | الإجراء |
|---|---|
| `No module named tensorflow` | فعّل بيئة Python 3.11 ثم ثبّت `requirements.txt` |
| `Missing model_0.keras` | تأكد من وجود ملفات `models/` بعد فك الضغط |
| `PCA expected 2560 features` | لا تستخدم نموذجاً/مصنفاً من إصدار مختلف |
| كاميرا الهاتف لا تعمل | اسمح بالكاميرا واستخدم رابط HTTPS |
| `InconsistentVersionWarning` من sklearn | استخدم النسخة المثبتة `scikit-learn==1.6.1` |
| تحميل النموذج بطيء | أول تحميل قد يكون أبطأ؛ افحص الذاكرة وموارد الخادم |
| `Failed to load joblib` | الملف قد يكون تالفاً أو غير موثوق؛ تحقق من SHA-256 |
