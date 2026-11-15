import os
import json
from google import genai
from core.translation import is_valid_georgian

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'timeout': 60000})
s = json.load(open('projects/video 1_376f458e/state.json', encoding='utf-8'))
batch = [{'id': x['id'], 'text': x['text']} for x in s['segments'][:15]]

prompt = f"""You are an expert film dubbing translator.
Translate the following dialogue lines into natural, conversational, spoken Georgian for film dubbing.
Also analyze the conversational context and detect whether each line is spoken by a 'male' or 'female' speaker.
Return ONLY a valid JSON list of objects:
[{{"id": 0, "translation": "...", "gender": "male"|"female"}}]
Strictly no markdown or explanations.

Input:
{json.dumps(batch, ensure_ascii=False)}
"""

resp = client.models.generate_content(model='gemini-3.5-flash-lite', contents=prompt)
raw = resp.text.strip()
if raw.startswith('```'):
    raw = raw.split('\n', 1)[1].rsplit('\n', 1)[0].strip()
parsed = json.loads(raw)
print(f'Successfully parsed {len(parsed)} items from Gemini!')
for item in parsed[:5]:
    b = item['translation'].encode('utf-8')
    print(f"ID {item['id']}: gender={item.get('gender')} | {b}")
