import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# إعدادات الصفحة
st.set_page_config(
    page_title="منظومة تدقيق عناوين الرسائل والأطاريح",
    page_icon="🎓",
    layout="centered"
)

# تخصيص الواجهة من اليمين إلى اليسار
st.markdown("""
    <style>
    body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        color: #1E3A8A;
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 15px;
        text-align: center;
        margin-bottom: 25px;
    }
    .card-match {
        background-color: #FFFBEB;
        border-right: 5px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .footer-note {
        font-size: 12px;
        color: #6B7280;
        text-align: center;
        margin-top: 40px;
        border-top: 1px solid #E5E7EB;
        padding-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# تنظيف الحروف العربية للفحص اللفظي
def clean_arabic_text(text):
    text = str(text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# قياس التشابه اللفظي
def lexical_similarity(text1, text2):
    t1_clean = clean_arabic_text(text1)
    t2_clean = clean_arabic_text(text2)
    return SequenceMatcher(None, t1_clean, t2_clean).ratio()

# تحميل النموذج وقاعدة البيانات وحفظها في الكاش
@st.cache_resource
def load_system():
    # نموذج لغوي يدعم العربية ومتعدد اللغات
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # قراءة الإكسيل
    df = pd.read_excel('titles.xlsx')
    
    # التعرف على عمود العناوين
    title_col = None
    for col in df.columns:
        if 'عنوان' in str(col):
            title_col = col
            break
    if not title_col:
        title_col = df.columns[0]
        
    titles_list = df[title_col].dropna().astype(str).str.strip().tolist()
    titles_list = [t for t in titles_list if len(t) > 5]
    
    # تحويل العناوين لمتجهات دلالية
    embeddings = model.encode(titles_list, convert_to_tensor=False)
    
    return model, titles_list, embeddings

# تحميل المنظومة
with st.spinner("جاري تهيئة المنظومة وقاعدة البيانات..."):
    try:
        model, database_titles, db_embeddings = load_system()
    except Exception as e:
        st.error("يرجى التأكد من وجود ملف 'titles.xlsx' في المستودع.")
        st.stop()

# رأس الصفحة
st.markdown('<div class="main-title">🎓 منظومة تدقيق عناوين الرسائل والأطاريح</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">أدخلي العنوان المقترح في الحقل أدناه لفحص مدى توفره أو تشابهه مع العناوين المسجلة سابقاً</div>', unsafe_allow_html=True)

# حقل الإدخال
proposed_title = st.text_area(
    label="العنوان المقترح:",
    placeholder="اكتبي العنوان المقترح هنا بالكامل...",
    height=100
)

# زر الفحص
if st.button("🔍 فحص العنوان المقترح", use_container_width=True):
    if not proposed_title or len(proposed_title.strip()) < 5:
        st.warning("يرجى كتابة عنوان متكامل للبدء بالفحص.")
    else:
        with st.spinner("جاري تحليل العنوان ومقارنة الدلالات والألفاظ..."):
            # الفحص بالمعنى
            input_embedding = model.encode([proposed_title.strip()], convert_to_tensor=False)
            semantic_scores = cosine_similarity(input_embedding, db_embeddings)[0]
            
            # الفحص اللفظي والدمج
            results = []
            for i, db_title in enumerate(database_titles):
                sem_score = float(semantic_scores[i])
                lex_score = lexical_similarity(proposed_title, db_title)
                
                # معادلة الدمج (65% للمعنى + 35% للألفاظ)
                final_score = (sem_score * 0.65) + (lex_score * 0.35)
                percentage = int(round(final_score * 100))
                
                # شرط الـ 60%
                if percentage >= 60:
                    results.append({
                        "title": db_title,
                        "score": percentage,
                        "sem_score": int(round(sem_score * 100)),
                        "lex_score": int(round(lex_score * 100))
                    })
            
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            st.write("---")
            
            if results:
                st.warning(f"⚠️ تنبيه: تم العثور على ({len(results)}) عناوين سابقة ذات تشابه ملحوظ (تجاوزت 60%).")
                
                for item in results[:5]:  # عرض أعلى 5 نتائج
                    st.markdown(f"""
                        <div class="card-match">
                            <strong style="color: #1E3A8A; font-size: 16px;">📖 العنوان المسجل:</strong><br>
                            <span style="font-size: 15px; font-weight: 500;">{item['title']}</span>
                            <hr style="margin: 8px 0; border: none; border-top: 1px dashed #D1D5DB;">
                            <span style="color: #B45309; font-weight: bold; font-size: 14px;">
                                📊 نسبة التطابق الإجمالية: {item['score']}%
                            </span>
                            <span style="color: #6B7280; font-size: 12px; margin-right: 15px;">
                                (تقارب المعنى: {item['sem_score']}% | تقارب اللفظ: {item['lex_score']}%)
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ العنوان متاح: لم يُعثر على عناوين سابقة ذات تشابه مسجلة في قاعدة البيانات بنسبة تزيد عن 60%.")

# التذييل
st.markdown('<div class="footer-note">📌 ملاحظة: هذا الفحص الآلي استرشادي وذكي، ويبقى الاعتماد النهائي خاضعاً لقرار اللجان العلمية في القسم والكلية.</div>', unsafe_allow_html=True)
