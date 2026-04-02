import os
import json
import re
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.genai as genai

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'api-practice', '.env'))
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("오류: GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    exit(1)

client = genai.Client(api_key=api_key)

app = Flask(__name__)

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


def clean_keywords(keywords, brand_name=None, existing_keywords=None):
    """키워드 정리: 특수문자 제거, 브랜드명 중복 필터, 필수키워드 중복 제거."""
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract_keywords():
    data = request.json
    brands = data.get('brands', [])

    if not brands:
        return jsonify({'error': '브랜드 정보를 입력해주세요.'}), 400

    all_existing = {}

    user_prompt = "아래 브랜드와 상품 목록에 대해 네이버 검색광고용 키워드를 추출해주세요.\n\n"
    for brand_info in brands:
        brand_name = brand_info.get('brand', '').strip()
        products = brand_info.get('products', [])
        existing_kw_raw = brand_info.get('existingKeywords', [])
        if not brand_name or not products:
            continue

        existing_kw = []
        for item in existing_kw_raw:
            existing_kw.extend(re.split(r'[;,\n]+', item))
        existing_clean = [re.sub(r'[^가-힣a-zA-Z0-9]', '', k).strip() for k in existing_kw if k.strip()]
        existing_clean = [k for k in existing_clean if k]
        all_existing[brand_name] = set(existing_clean)

        user_prompt += f"브랜드: {brand_name}\n상품목록:\n"
        for p in products:
            p = p.strip()
            if p:
                user_prompt += f"- {p}\n"

        if existing_clean:
            user_prompt += f"\n필수키워드 (이미 등록됨, 결과에서 제외하되 방향성 참고):\n"
            for ek in existing_clean:
                user_prompt += f"- {ek}\n"

        user_prompt += "\n"

    last_error = None

    for model_name in FALLBACK_MODELS:
        success = False
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
                    last_error = Exception('AI 응답 파싱 실패')
                    break

                result = json.loads(json_match.group())

                for brand_name in result:
                    existing = all_existing.get(brand_name, set())
                    result[brand_name] = clean_keywords(result[brand_name], brand_name, existing)

                return jsonify({'result': result, 'model': model_name})

            except Exception as e:
                last_error = e
                error_str = str(e)
                app.logger.warning(f"모델 {model_name} 시도 {attempt+1} 실패: {error_str[:120]}")
                if is_retryable_error(error_str) and attempt < 1:
                    time.sleep(5)
                    continue
                else:
                    break

        app.logger.warning(f"모델 {model_name} 최종 실패, 다음 모델 시도")

    return jsonify({
        'error': f'모든 AI 모델이 일시적으로 사용 불가합니다. 잠시 후 다시 시도해주세요.\n(상세: {str(last_error)[:200]})'
    }), 503


if __name__ == '__main__':
    app.run(debug=True, port=5000)
