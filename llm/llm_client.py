# llm/llm_client.py
import requests
import json
import time
import logging

logger = logging.getLogger(__name__)

APPRAISAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "AppraisalAnalysis",
        "schema": {
            "type": "object",
            "properties": {
                # 1. Overall Summary
                "comparative_summary": {
                    "type": "string",
                    "description": "Objective summary of key events in Video A vs Video B."
                },
                # 2. Analyze each appraisal dimension
                "novelty": {
                    "type": "object",
                    "properties": {
                        # Objective description -> Reasoning -> Prediction
                        "description": {"type": "string", "description": "Objective description of visual events related to Novelty."},
                        "rationale": {"type": "string", "description": "Logical reasoning comparing A and B based on the description."},
                        "predicted_winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["description", "rationale", "predicted_winner"],
                    "additionalProperties": False
                },
                "goal_relevance": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Objective description of visual events related to Goal Relevance."},
                        "rationale": {"type": "string", "description": "Logical reasoning comparing A and B based on the description."},
                        "predicted_winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["description", "rationale", "predicted_winner"],
                    "additionalProperties": False
                },
                "outcome_probability": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Objective description of visual events related to Outcome Probability."},
                        "rationale": {"type": "string", "description": "Logical reasoning comparing A and B based on the description."},
                        "predicted_winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["description", "rationale", "predicted_winner"],
                    "additionalProperties": False
                },
                "goal_conduciveness": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Objective description of visual events related to Goal Conduciveness."},
                        "rationale": {"type": "string", "description": "Logical reasoning comparing A and B based on the description."},
                        "predicted_winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["description", "rationale", "predicted_winner"],
                    "additionalProperties": False
                },
                "urgency": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Objective description of visual events related to Urgency."},
                        "rationale": {"type": "string", "description": "Logical reasoning comparing A and B based on the description."},
                        "predicted_winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["description", "rationale", "predicted_winner"],
                    "additionalProperties": False
                },
                "coping_potential": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Objective description of visual events related to Coping Potential."},
                        "rationale": {"type": "string", "description": "Logical reasoning comparing A and B based on the description."},
                        "predicted_winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["description", "rationale", "predicted_winner"],
                    "additionalProperties": False
                }
            },
            "required": [
                "comparative_summary",
                "novelty", "goal_relevance", "outcome_probability",
                "goal_conduciveness", "urgency", "coping_potential"
            ],
            "additionalProperties": False
        }
    }
}


def get_appraisal_analysis(api_key, system_prompt, user_prompt, base64_a, base64_b, model="google/gemini-2.5-flash",
                           retries=3):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "# Video A"},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{base64_a}"}},
                    {"type": "text", "text": "# Video B"},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{base64_b}"}}
                ]
            }
        ],
        "response_format": APPRAISAL_SCHEMA,
        "plugins": [
            {"id": "response-healing"}
        ]
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                # Return parsed JSON object
                return json.loads(content)
            else:
                logger.warning(f"Empty response from API: {data}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"API Attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None

    return None