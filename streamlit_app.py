import os
import json
import re
import time
import streamlit as st
import google.genai as genai
import io

st.set_page_config(
    page_title="네이버 검색광고 키워드 추출기",
    page_icon="🔍",
    layout="wide",
)

FALLBACK_MODELS = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-flash']

FEW_SHOT_EXAMPLES = """
[예시 1]
브랜드: 마인드브릿지
상품: [올데이] 옥스포드 오버핏 셔츠_4colors / [올데이] 데님 오버핏 셔츠_3colors / [COUTURE] 와이드 원턱 밴딩 팬츠_2colors / 테이퍼드 밴딩 팬츠_2colors / 스웻하프팬츠_2colors / [올데이] 수피마 오버핏 셔츠_3colors
추출 (27개): 마인드브릿지데님셔츠, 마인드브릿지데님오버핏셔츠, 마인드브릿지세미와이드, 마인드브릿지세미와이드밴딩팬츠, 마인드브릿지수피마, 마인드브릿지수피마셔츠, 마인드브릿지수피마오버핏셔츠, 마인드브릿지스웻, 마인드브릿지스웻팬츠, 마인드브릿지스웻하프팬츠, 마인드브릿지슬랙스, 마인드브릿지오버핏데님셔츠, 마인드브릿지오버핏수피마셔츠, 마인드브릿지옴브레체크, 마인드브릿지옴브레체크셔츠, 마인드브릿지와이드슬랙스, 마인드브릿지와이드원턱팬츠, 마인드브릿지와이드팬츠, 마인드브릿지원턱팬츠, 마인드브릿지하프팬츠

[예시 2]
브랜드: 아이오페
상품: 슈퍼바이탈 아이크림 25ml / 슈퍼바이탈 크림 세트 / 슈퍼바이탈 2종 (소프너+에멀젼) 세트 / 비타민C 25% 항산화 토닝앰플 2개 / UV 쉴드 선 프로텍터 2개 / UV 쉴드 톤업 선 2개 / 레티놀 엑스퍼트 0.1% 기획세트
추출 (16개): 아이오페, 아이오페레티놀, 아이오페레티놀엑스퍼트, 아이오페비타민, 아이오페선크림, 아이오페소프너, 아이오페슈퍼바이탈, 아이오페슈퍼바이탈크림, 아이오페아이크림, 아이오페앰플, 아이오페에멀전, 아이오페에센스, 아이오페오일, 아이오페크림, 아이오페토닝앰플, 아이오페톤업선크림
"""

SYSTEM_PROMPT = """네이버 검색광고 키워드 추출 전문가. 브랜드+상품목록 → 검색광고용 키워드 추출.

## 규칙
1. 형식: 브랜드명+핵심단어 결합. 띄어쓰기/특수문자 절대 불포함.
2. 상품명에서 핵심 단어 추출 (대괄호, 색상, 용량, 수량, 할인율 제거. 영문→한글 변환/제거)
3. 브랜드+단어1, 브랜드+단어1+단어2 등 다양한 조합 생성. 연관 단어 포함 (롱슬리브→긴팔 등)
4. 브랜드 단독 키워드 포함 (예: 아이오페)
5. **브랜드당 20~30개** 키워드만 추출 (핵심적이고 검색 가능성 높은 키워드 우선)
6. 중복 제거, 가나다순 정렬
7. 필수키워드가 제공되면: 결과에서 제외하되, 패턴/방향성을 참고하여 확장 추출

## 절대 금지 (매우 중요)
- **브랜드명 중복 금지**: 키워드 안에 브랜드명이 두 번 이상 들어가면 안 됩니다.
  - 나쁜 예: 디올디올쇼, 아이오페아이오페크림 → 브랜드명이 두 번 반복됨
  - 좋은 예: 디올립글로우, 아이오페크림 → 브랜드명 1번 + 상품키워드
- **의미 없는 단어 조합 금지**: 브랜드명 뒤에 붙이는 단어는 반드시 소비자가 실제 검색할 만한 **상품 카테고리/종류**(셔츠, 크림, 자켓, 향수, 립스틱 등), **소재**(울, 데님, 수피마 등), **라인명**(슈퍼바이탈, 레티놀 등), 또는 이들의 조합이어야 합니다.
  - 나쁜 예: 마리끌레르버튼, 마리끌레르베이직, 마리끌레르오픈 → 단독으로 의미 없는 수식어
  - 좋은 예: 마리끌레르롱슬리브, 마리끌레르울자켓, 마리끌레르체크셔츠 → 검색 가능한 상품 카테고리 포함

## 예시
""" + FEW_SHOT_EXAMPLES + """
## 응답 형식 (JSON만, 다른 텍스트 없이)
```json
{"브랜드명1": ["키워드1", "키워드2", ...]}
```
"""


def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", "")


def clean_keywords(keywords, brand_name=None, existing_keywords=None):
    existing = set(existing_keywords or [])
    brand_lower = brand_name.lower().replace(' ', '') if brand_name else None
    cleaned = []
    for kw in keywords:
        kw = re.sub(r'[^가-힣a-zA-Z0-9]', '', kw)
        kw = kw.strip()
        if not kw or len(kw) <= 1 or kw in existing:
            continue
        if brand_lower:
            kw_lower = kw.lower()
            if kw_lower.startswith(brand_lower):
                remainder = kw_lower[len(brand_lower):]
                if remainder.startswith(brand_lower):
                    continue
        cleaned.append(kw)
    return sorted(set(cleaned))


def is_retryable_error(error_str):
    retryable = ['429', 'RESOURCE_EXHAUSTED', '503', 'UNAVAILABLE', '500', 'INTERNAL']
    return any(code in error_str for code in retryable)


def parse_keyword_text(text):
    return [k.strip() for k in re.split(r'[;,\n]+', text) if k.strip()]


def extract_keywords_api(brands, api_key):
    client = genai.Client(api_key=api_key)
    all_existing = {}

    user_prompt = "아래 브랜드와 상품 목록에 대해 네이버 검색광고용 키워드를 추출해주세요.\n\n"
    for brand_info in brands:
        brand_name = brand_info['brand']
        products = brand_info['products']
        existing_kw = brand_info.get('existing', [])

        existing_clean = [re.sub(r'[^가-힣a-zA-Z0-9]', '', k).strip() for k in existing_kw if k.strip()]
        existing_clean = [k for k in existing_clean if k]
        all_existing[brand_name] = set(existing_clean)

        user_prompt += f"브랜드: {brand_name}\n상품목록:\n"
        for p in products:
            user_prompt += f"- {p}\n"

        if existing_clean:
            user_prompt += f"\n필수키워드 (이미 등록됨, 결과에서 제외하되 방향성 참고):\n"
            for ek in existing_clean:
                user_prompt += f"- {ek}\n"
        user_prompt += "\n"

    last_error = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config={
                        'system_instruction': SYSTEM_PROMPT,
                        'temperature': 0.3,
                    }
                )
                response_text = response.text.strip()
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if not json_match:
                    last_error = "AI 응답 파싱 실패"
                    break

                result = json.loads(json_match.group())
                for bn in result:
                    existing = all_existing.get(bn, set())
                    result[bn] = clean_keywords(result[bn], bn, existing)
                return result, model_name

            except Exception as e:
                last_error = str(e)
                if is_retryable_error(last_error) and attempt < 1:
                    time.sleep(5)
                    continue
                else:
                    break

    raise Exception(f"모든 AI 모델이 일시적으로 사용 불가합니다. 잠시 후 다시 시도해주세요.\n({last_error[:200] if last_error else ''})")


# ── UI ──

st.markdown("""
<style>
    .block-container { max-width: 1100px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    .brand-card {
        background: white;
        border: 1px solid #e0e4e8;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kw-tag {
        display: inline-block;
        background: #f8f9fa;
        border: 1px solid #e0e4e8;
        padding: 6px 14px;
        border-radius: 6px;
        margin: 3px;
        font-size: 14px;
    }
    .stat-box {
        background: white;
        border: 1px solid #e0e4e8;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-box .val { font-size: 28px; font-weight: 800; color: #03c75a; }
    .stat-box .lbl { font-size: 13px; color: #666; }
    .result-header {
        background: linear-gradient(135deg, #e8f8ef, #f0fdf5);
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #03c75a, #00b843); color:white; padding:28px 0 24px; text-align:center; border-radius:0 0 16px 16px; margin: -1rem -1rem 1.5rem -1rem; box-shadow: 0 4px 20px rgba(3,199,90,0.25);">
    <h1 style="font-size:26px; margin:0 0 6px 0;">네이버 검색광고 키워드 추출기</h1>
    <p style="font-size:14px; opacity:0.9; margin:0;">브랜드와 상품명을 입력하면 AI가 검색광고용 키워드를 자동 추출합니다</p>
</div>
""", unsafe_allow_html=True)

api_key = get_api_key()
if not api_key:
    st.warning("Gemini API Key가 설정되지 않았습니다. 아래에 입력하거나, Streamlit Secrets에 `GEMINI_API_KEY`를 등록하세요.")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key:
        st.stop()

st.info("**사용법:** 브랜드명을 입력하고, 상품명을 한 줄에 하나씩 입력하세요. 상품 개수 제한 없이 자유롭게 등록 가능합니다. 이미 운영 중인 필수키워드가 있다면 함께 입력하면 중복 없이 확장된 키워드를 추출합니다.")

if 'brand_count' not in st.session_state:
    st.session_state.brand_count = 1

col_add, col_remove = st.columns([1, 1])
with col_add:
    if st.button("➕ 브랜드 추가", use_container_width=True):
        st.session_state.brand_count += 1
        st.rerun()
with col_remove:
    if st.session_state.brand_count > 1:
        if st.button("➖ 마지막 브랜드 삭제", use_container_width=True):
            st.session_state.brand_count -= 1
            st.rerun()

brands_data = []
for i in range(st.session_state.brand_count):
    with st.container():
        st.markdown(f"### 브랜드 {i+1}")
        brand_name = st.text_input(
            "브랜드명",
            key=f"brand_{i}",
            placeholder="예: 마리끌레르",
            label_visibility="collapsed",
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**상품 목록**")
            products_text = st.text_area(
                "상품 목록",
                key=f"products_{i}",
                height=150,
                placeholder="상품명을 한 줄에 하나씩 입력\n\n예시:\n울 버튼 핸드메이드 자켓\n아치 로고 래글런 롱슬리브",
                label_visibility="collapsed",
            )
        with right:
            st.markdown("**필수키워드** *(선택)*")
            existing_text = st.text_area(
                "필수키워드",
                key=f"existing_{i}",
                height=150,
                placeholder="이미 등록된 키워드를 붙여넣기\n(세미콜론, 쉼표, 줄바꿈 모두 지원)\n\n예시:\n불가리;불가리퍼퓸;불가리퍼퓸오드퍼퓸",
                label_visibility="collapsed",
            )

        products = [p.strip() for p in products_text.split('\n') if p.strip()]
        existing = parse_keyword_text(existing_text) if existing_text else []

        prod_count = len(products)
        ek_count = len(existing)
        st.caption(f"상품 {prod_count}개 | 필수키워드 {ek_count}개")

        brands_data.append({
            'brand': brand_name.strip() if brand_name else '',
            'products': products,
            'existing': existing,
        })

    st.divider()

if st.button("🔍 키워드 추출", type="primary", use_container_width=True):
    valid_brands = [b for b in brands_data if b['brand'] and b['products']]

    if not valid_brands:
        st.error("브랜드명과 상품을 최소 1개 이상 입력해주세요.")
    else:
        with st.spinner("AI가 최적의 키워드를 분석하고 있습니다..."):
            try:
                result, model_used = extract_keywords_api(valid_brands, api_key)
                st.session_state.last_result = result
                st.session_state.model_used = model_used
            except Exception as e:
                st.error(str(e))
                st.stop()

if 'last_result' in st.session_state:
    result = st.session_state.last_result

    st.markdown("---")
    st.markdown("## 추출 결과")

    brand_names = list(result.keys())
    total_kw = sum(len(result[b]) for b in brand_names)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="val">{len(brand_names)}</div><div class="lbl">브랜드 수</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="val">{total_kw}</div><div class="lbl">총 키워드 수</div></div>', unsafe_allow_html=True)

    st.markdown("")

    all_keywords_text = ""
    csv_data = "\ufeff브랜드,키워드\n"

    for brand_name in brand_names:
        keywords = result[brand_name]
        all_keywords_text += "\n".join(keywords) + "\n"

        for kw in keywords:
            csv_data += f"{brand_name},{kw}\n"

        st.markdown(f'<div class="result-header"><strong>{brand_name}</strong><span style="background:#03c75a;color:white;padding:3px 12px;border-radius:12px;font-size:13px;">{len(keywords)}개</span></div>', unsafe_allow_html=True)

        tags_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in keywords)
        st.markdown(f'<div style="padding:8px 0 16px;">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown("---")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "📋 전체 키워드 복사용 (TXT)",
            data=all_keywords_text,
            file_name="naver_keywords.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📊 CSV 다운로드",
            data=csv_data,
            file_name="naver_keywords.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl3:
        try:
            import openpyxl
            from openpyxl import Workbook
            wb = Workbook()
            wb.remove(wb.active)

            for bn in brand_names:
                ws = wb.create_sheet(title=bn[:31])
                ws.append(["키워드"])
                ws.column_dimensions['A'].width = 40
                for kw in result[bn]:
                    ws.append([kw])

            ws_all = wb.create_sheet(title="전체")
            ws_all.append(["브랜드", "키워드"])
            ws_all.column_dimensions['A'].width = 20
            ws_all.column_dimensions['B'].width = 40
            for bn in brand_names:
                for kw in result[bn]:
                    ws_all.append([bn, kw])

            buf = io.BytesIO()
            wb.save(buf)
            st.download_button(
                "📥 Excel 다운로드",
                data=buf.getvalue(),
                file_name="naver_keywords.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            st.caption("Excel 다운로드에는 openpyxl 패키지가 필요합니다.")
