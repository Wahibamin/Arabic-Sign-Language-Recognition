"""ArSL V5: formal Arabic interactive demonstration built with Gradio."""

from __future__ import annotations

import html
import logging
import os

import gradio as gr

from src.arsl.predictor import get_predictor, validate_assets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STYLES = """
:root {--ink:#102b3c;--teal:#117d78;--muted:#687f8a;--paper:#f6fafb;--line:#dae6e8}
.gradio-container {font-family:Tahoma,'Segoe UI',Arial,sans-serif !important;direction:rtl;color:var(--ink);max-width:1190px!important;margin:0 auto!important}
body {background:linear-gradient(165deg,#eff7f7 0%,#ffffff 58%,#f5f9fb 100%)!important}
#hero {background:linear-gradient(110deg,#102f42,#146d72);padding:32px 34px;border-radius:22px;color:#fff;direction:rtl;box-shadow:0 16px 40px rgba(19,63,75,.13)}
#hero h1{font-size:clamp(1.55rem,3.4vw,2.35rem);font-weight:800;line-height:1.7;margin:8px 0}
#hero p{color:#ddf4f2;line-height:1.95;font-size:1rem;max-width:850px}
#hero .eyebrow{font-size:.8rem;letter-spacing:.8px;color:#abe8d9;font-weight:700}
#stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0 20px}
#stats .stat{border:1px solid #d7e7e8;border-radius:14px;background:#fff;padding:15px 18px;box-shadow:0 5px 20px rgba(16,56,65,.035)}
#stats strong{font-size:1.7rem;display:block;color:#166f74;direction:ltr;text-align:right}
#stats small{color:#607985;font-size:.8rem;line-height:1.7}
#instructions, #method {direction:rtl;line-height:2;color:#324b5a}
#uploadPanel, #outPanel {border:1px solid #d9e7e8;border-radius:18px;background:#fff!important;padding:16px;box-shadow:0 6px 30px rgba(24,66,76,.045)}
#uploadPanel>*, #outPanel>*{direction:rtl}
.pred-card{direction:rtl;background:#fff;border:1px solid #dce8ea;border-radius:14px;padding:18px 20px;min-height:295px;color:#183541}
.pred-card h3{font-size:1rem;color:#53717b;margin:0 0 15px}
.pred-head{display:flex;gap:14px;align-items:center;margin-bottom:14px}
.pred-letter{font-size:2.7rem;font-weight:800;min-width:72px;height:72px;background:#e8f7f2;border-radius:17px;display:flex;align-items:center;justify-content:center;color:#116a68}
.pred-main{font-weight:800;font-size:1.15rem}
.pred-sub{font-size:.85rem;color:#768994}
.pred-row{display:flex;gap:12px;align-items:center;margin:10px 0;font-size:.87rem}
.pred-row .name{min-width:65px;font-weight:750}
.pred-row .track{height:9px;background:#e9f1f2;border-radius:12px;flex:1;overflow:hidden}
.pred-row .fill{height:100%;background:linear-gradient(90deg,#0c837c,#51bfa5);border-radius:12px}
.pred-row .pct{direction:ltr;min-width:52px;color:#435d68;font-variant-numeric:tabular-nums;text-align:left}
.caution{background:#fff9ee;border:1px solid #f1e3c6;color:#755c36;border-radius:11px;padding:9px 12px;font-size:.8rem;line-height:1.8;margin-top:14px}
.placeholder{color:#748a92;text-align:center;padding:62px 9px;line-height:2}
.error{border-color:#f2c8c8;color:#8e2c35;background:#fff6f6}
#foot{margin:16px 0 28px;color:#6d8389;font-size:.8rem;text-align:center;direction:rtl;line-height:1.9}
@media(max-width:640px){#hero{padding:23px 18px}#stats{grid-template-columns:repeat(3,1fr);gap:6px}#stats .stat{padding:10px 8px}#stats strong{font-size:1.2rem}#stats small{font-size:.67rem}#uploadPanel,#outPanel{padding:8px}}
"""

HERO = """<header id='hero'><div class='eyebrow'>ARSL • ACADEMIC DEMONSTRATION • V5</div>
<h1>التعرّف على حروف لغة الإشارة العربية</h1>
<p>نظام تجريبي لتصنيف إشارات الأبجدية العربية الثابتة من الصور. يدمج ميزات شبكتين عميقتين
مع مصنّف SVM للوصول إلى قرار تصنيفي موحّد.</p></header>"""

STATS = """<div id='stats'>
<div class='stat'><strong>31</strong><small>فئة تصنيف</small></div>
<div class='stat'><strong>88.31%</strong><small>دقة الاختبار الاستكشافية المسجلة</small></div>
<div class='stat'><strong>V5</strong><small>النموذج البحثي المعتمد</small></div>
</div>"""


def empty_card() -> str:
    return "<section class='pred-card'><h3>نتيجة التصنيف</h3><div class='placeholder'>اختر صورة واضحة لليد ثم اضغط «تحليل الإشارة» لعرض أفضل التوقعات.</div></section>"


def render_prediction(result) -> str:
    letter = html.escape(result.best_arabic)
    english = html.escape(result.best_class)
    bars = []
    for cls, arabic, score in result.top_five:
        percentage = max(0, min(100, score * 100))
        bars.append(
            "<div class='pred-row'>"
            f"<span class='name'>{html.escape(arabic)}</span>"
            f"<div class='track'><div class='fill' style='width:{percentage:.2f}%'></div></div>"
            f"<span class='pct'>{percentage:.1f}%</span>"
            "</div>"
        )
    warning = (
        "<div class='caution'>تنبيه: تقاربت نتائج الفئات؛ يُفضّل صورة أوضح بإضاءة جيدة.</div>"
        if result.confidence < 0.50
        else ""
    )
    return (
        "<section class='pred-card'>"
        "<h3>نتيجة التصنيف</h3>"
        f"<div class='pred-head'><div class='pred-letter'>{letter}</div>"
        f"<div><div class='pred-main'>الإشارة الأقرب: {letter}</div>"
        f"<div class='pred-sub'>{english} · درجة ترجيح {result.confidence * 100:.1f}%</div>"
        "</div></div><h3>الفئات الخمس الأقرب</h3>"
        + "".join(bars)
        + warning
        + "<div class='caution'>الدرجات تقديرية وليست احتمالات مُعايرة؛ النظام مخصّص لإشارات الحروف الثابتة فقط.</div>"
        "</section>"
    )


def classify(image):
    if image is None:
        return empty_card(), "يرجى رفع صورة أو التقاط صورة من الكاميرا أولاً."
    try:
        result = get_predictor().predict(image)
        return render_prediction(result), "اكتمل التحليل. تعرض الواجهة أفضل خمس فئات بحسب درجة الترجيح."
    except Exception:
        logger.exception("Model inference failed")
        return (
            "<section class='pred-card error'><h3>تعذر التحليل</h3>"
            "<p>تحقق من وجود ملفات النموذج ومن صيغة الصورة، ثم أعد المحاولة.</p></section>",
            "تعذّر إكمال التنبؤ. راجع سجلات الخادم لمزيد من التفاصيل.",
        )


def make_app() -> gr.Blocks:
    with gr.Blocks(title="ArSL V5 | التعرف على حروف الإشارة العربية") as demo:
        gr.HTML(HERO)
        gr.HTML(STATS)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=325, elem_id="uploadPanel"):
                gr.Markdown("### إدخال صورة الإشارة")
                upload = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="صورة اليد",
                    height=330,
                    image_mode="RGB",
                )
                with gr.Row():
                    run_btn = gr.Button("تحليل الإشارة", variant="primary", size="lg")
                    reset_btn = gr.Button("مسح", variant="secondary")
                message = gr.Markdown("اختر إشارة واحدة بوضوح داخل إطار الصورة.")
            with gr.Column(scale=1, min_width=325, elem_id="outPanel"):
                output = gr.HTML(value=empty_card())
        run_btn.click(fn=classify, inputs=[upload], outputs=[output, message])
        reset_btn.click(
            fn=lambda: (None, empty_card(), "اختر إشارة واحدة بوضوح داخل إطار الصورة."),
            outputs=[upload, output, message],
        )
        with gr.Accordion("المنهج العلمي المستخدم", open=False):
            gr.Markdown(
                "**سير المعالجة:** صورة RGB → MobileNetV2 (160×160) + EfficientNetV2B0 "
                "(192×192) → متجهات ميزات 1280 + 1280 → L2 Normalization → PCA (128) "
                "→ SVM/RBF → دمج (50% من توقعات CNN و50% من SVM). "
                "لم يتطلب الإصدار V5 إعادة تدريب الشبكتين أثناء بناء مصنّف SVM."
            )
        gr.HTML(
            "<footer id='foot'>مشروع أكاديمي في الرؤية الحاسوبية والتعلم العميق. "
            "دقة 88.31% مسجلة على مجموعة اختبار سبق استخدامها في تجارب سابقة، "
            "ولا تمثل ضماناً للدقة على مستخدمين جدد. "
            "لا تُرفع صور المستخدم إلى مستودع GitHub.</footer>"
        )
    return demo


app = make_app()

if __name__ == "__main__":
    missing = validate_assets()
    if missing:
        raise SystemExit("ملفات النماذج المطلوبة غير موجودة: " + ", ".join(missing))
    port = int(os.environ.get("PORT", "7860"))
    app.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=port, share=False,
        css=STYLES, theme=gr.themes.Soft(primary_hue="teal"),
    )
