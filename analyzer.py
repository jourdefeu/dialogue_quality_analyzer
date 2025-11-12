import json
import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_dialogues(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """
    нормализация текста перед отправкой в LLM.
    """
    text = text.lower()
    text = re.sub(r'[()]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def analyze_dialogue(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    messages — список сообщений формата:
    [{"sender": "Менеджер", "timestamp": "...", "text": "..."}]
    Возвращает результат анализа по разделам A и B.
    """
    dialogue_text = "\n".join([f"{m['sender']}: {m['text']}" for m in messages])

    analyzer = {
        "section_a": a_anchors(dialogue_text),
        "section_b": b_objections(dialogue_text)
    }

    for obj in analyzer["section_b"].get("objections_found", []):
        obj["objection_type"] = obj["objection_type"].capitalize()
        obj["manager_actions"] = [
            action[0].upper() + action[1:] if action else action
            for action in obj.get("manager_actions", [])
        ]

    return analyzer

def a_anchors(msgs):

    prompt = f"""
Проанализируй диалог между менеджером и клиентом.
Определи, какие из пяти ключевых тем обсуждал менеджер.

Диалог: {msgs}

Ключевые темы: 
1. Цели и KPI/метрики успеха (целевые KPI (CPA/ROMI/лиды), критерии успеха)
2. Текущие источники трафика (где рекламируются сейчас, с кем работали)
3. Бюджет (планируемый бюджет, модель оплаты, минимальные депозиты)
4. URL/артефакты (ссылки на лендинги, креативы, материалы)
5. Ожидания от партнёра (что важно клиенту, какие условия критичны)

Формат раздела (пример):
{{
"info_anchors_found": [
    "Бюджет",
    "Текущие источники трафика",
    "Ожидания от партнёра"
],
"info_anchors_missing": [
    "Цели и KPI/метрики успеха",
    "URL/артефакты"
]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "LLM вернул невалидный JSON", "raw_output": text}

def b_objections(msgs):

    prompt = f"""
Проанализируй диалог между менеджером и клиентом.
Найди все возражения клиента и оцени, как менеджер их отработал.

Диалог: {msgs}

Типы возражений (objection_type):
1. Финансовые ограничения (триггеры этого возражения - "дорого", "нет бюджета", "высокая комиссия")
2. Невыгодные условия сотрудничества (триггеры этого возражения - "не устраивают условия", "высокий депозит")
3. Потеря в пользу конкурента (триггеры этого возражения - "у конкурента выгоднее", "другое агентство")

Для КАЖДОГО возражения верни объект:
{{
"objection_type": "тип возражения",
"client_quote": "цитата клиента",
"manager_handled": true/false,
"manager_actions": ["список конкретных действий менеджера (отработка возражений)"]
}}
Помести его в формат раздела. Всего будет ТРИ таких объекта в разделе (все они). Типы возражений НЕ ДОЛЖНЫ повторяться.

Формат раздела (пример):
{{
"objections_found": [ ... ]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "LLM вернул невалидный JSON", "raw_output": text}



if __name__ == "__main__":
    data = load_dialogues("dialogues_sample.json")
    results = []

    for i, dialogue in enumerate(data, 1):
        logger.info(f"Анализ диалога {i}/{len(data)}...")
        res = analyze_dialogue(dialogue["messages"])
        results.append({
            "dialogue_id": dialogue.get("dialogue_id", f"dlg_{i}"),
            "analysis": res
        })

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("Анализ завершён. Результаты сохранены в analysis_results.json")