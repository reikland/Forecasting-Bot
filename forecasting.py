import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ForecastResult:
    question_type: str
    forecasts: List[str]
    judge_feedback: List[str]
    supreme_decision: str


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, messages: List[Dict[str, Any]]) -> str:
        payload = {"model": self.model, "messages": messages}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OPENROUTER_URL, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()


def classification_prompt(title: str, context: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Classify the forecasting question as exactly one of NUMERIC, BINARY, or MCQ. "
                "Use only the provided title and context. Reply with the label only."
            ),
        },
        {
            "role": "user",
            "content": f"Title: {title}\nContext: {context}",
        },
    ]


def forecaster_prompt(question_type: str, title: str, context: str) -> List[Dict[str, str]]:
    task_description = {
        "NUMERIC": "Provide a single numeric forecast (number only).",
        "BINARY": "Provide the probability that the answer is YES as a percentage between 0 and 100.",
        "MCQ": "Provide probabilities for each option, ensuring they sum to 100%.",
    }[question_type]

    return [
        {
            "role": "system",
            "content": (
                "You are an independent forecaster. "
                f"{task_description} Do not explain your answer."
            ),
        },
        {
            "role": "user",
            "content": f"Question type: {question_type}\nTitle: {title}\nContext: {context}",
        },
    ]


def judge_prompt(question_type: str, title: str, context: str, forecasts: List[str]) -> List[Dict[str, str]]:
    forecast_list = "\n".join(f"Forecaster {i+1}: {forecast}" for i, forecast in enumerate(forecasts))
    return [
        {
            "role": "system",
            "content": (
                "You are a judge evaluating multiple forecasts. "
                "Assess reasoning quality, identify inconsistencies or outliers, and suggest adjustments. "
                "Do not provide a final forecast."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {question_type}\nTitle: {title}\nContext: {context}\n"
                f"Forecasts:\n{forecast_list}"
            ),
        },
    ]


def supreme_prompt(
    question_type: str,
    title: str,
    context: str,
    forecasts: List[str],
    judge_feedback: List[str],
) -> List[Dict[str, str]]:
    forecast_list = "\n".join(f"Forecaster {i+1}: {forecast}" for i, forecast in enumerate(forecasts))
    feedback_text = "\n".join(f"Judge {i+1}: {feedback}" for i, feedback in enumerate(judge_feedback))
    return [
        {
            "role": "system",
            "content": (
                "You are the Supreme Judge. Harmonize the forecasts using judge feedback. "
                "Respond in two short paragraphs: first explain reasoning, second present the final forecast. "
                "Use plain text only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {question_type}\nTitle: {title}\nContext: {context}\n"
                f"Forecasts:\n{forecast_list}\n"
                f"Judge feedback:\n{feedback_text}"
            ),
        },
    ]


async def run_pipeline(api_key: str, model: str, title: str, context: str) -> ForecastResult:
    client = LLMClient(api_key, model)

    question_type = (await client.complete(classification_prompt(title, context))).upper()

    if question_type not in {"NUMERIC", "BINARY", "MCQ"}:
        question_type = "NUMERIC"

    forecasts_tasks = [
        asyncio.create_task(client.complete(forecaster_prompt(question_type, title, context)))
        for _ in range(4)
    ]
    forecasts = [await task for task in forecasts_tasks]

    judge_tasks = [
        asyncio.create_task(client.complete(judge_prompt(question_type, title, context, forecasts)))
        for _ in range(2)
    ]
    judge_feedback = [await task for task in judge_tasks]

    supreme_decision = await client.complete(
        supreme_prompt(question_type, title, context, forecasts, judge_feedback)
    )

    return ForecastResult(
        question_type=question_type,
        forecasts=forecasts,
        judge_feedback=judge_feedback,
        supreme_decision=supreme_decision,
    )
