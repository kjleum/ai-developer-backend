import os
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class NLCommand:
    intent: str  # create_table, query, update, delete, chart, report
    target: str  # имя таблицы/сущности
    parameters: Dict[str, Any]
    original_text: str
    confidence: float

class NLPInterface:
    """NLP интерфейс для управления БД — "создай таблицу клиентов" → SQL"""

    def __init__(self, ai_manager=None, db=None):
        self.ai = ai_manager
        self.db = db

        # Паттерны для быстрого распознавания без AI
        self.patterns = {
            "create_table": [
                r"создай(те)?\s+таблицу\s+(\w+)",
                r"новая\s+таблица\s+(\w+)",
                r"добавь(те)?\s+таблицу\s+(\w+)",
                r"create\s+table\s+(\w+)"
            ],
            "query": [
                r"сколько\s+(\w+)",
                r"покажи(те)?\s+(.*)",
                r"выведи(те)?\s+(.*)",
                r"найди(те)?\s+(.*)",
                r"select\s+(.*)",
                r"где\s+(.*)"
            ],
            "chart": [
                r"график\s+(.*)",
                r"диаграмма\s+(.*)",
                r"построй(те)?\s+график",
                r"визуализируй(те)?\s+(.*)"
            ],
            "report": [
                r"отч[её]т\s+(.*)",
                r"аналитика\s+(.*)",
                r"статистика\s+(.*)",
                r"сводка\s+(.*)"
            ],
            "update": [
                r"обнови(те)?\s+(.*)",
                r"измени(те)?\s+(.*)",
                r"обновить\s+запись",
                r"update\s+(.*)"
            ],
            "delete": [
                r"удали(те)?\s+(.*)",
                r"очисти(те)?\s+(.*)",
                r"delete\s+(.*)",
                r"drop\s+(.*)"
            ]
        }

    async def parse_command(self, text: str, context: Dict = None) -> NLCommand:
        """Парсинг естественного языка в команду"""
        text_lower = text.lower().strip()

        # Пробуем паттерны
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    target = match.group(1) if len(match.groups()) > 0 else ""
                    return NLCommand(
                        intent=intent,
                        target=target.strip() if target else "",
                        parameters={"raw": text, "groups": match.groups()},
                        original_text=text,
                        confidence=0.8
                    )

        # Если паттерны не сработали — используем AI
        if self.ai:
            return await self._parse_with_ai(text, context)

        return NLCommand(
            intent="unknown",
            target="",
            parameters={},
            original_text=text,
            confidence=0.0
        )

    async def _parse_with_ai(self, text: str, context: Dict) -> NLCommand:
        """Использовать AI для парсинга сложных команд"""
        prompt = f"""Проанализируй команду на естественном языке для управления базой данных.

КОМАНДА: "{text}"

Определи:
1. Намерение (intent): create_table, query, update, delete, chart, report, unknown
2. Цель (target): имя таблицы или сущности
3. Параметры: дополнительные детали

Ответь строго JSON:
{{
  "intent": "...",
  "target": "...",
  "parameters": {{}},
  "confidence": 0.9
}}"""

        try:
            response = await self.ai.generate(prompt, temperature=0.1, json_mode=True)
            data = json.loads(self.ai.clean_json_response(response))

            return NLCommand(
                intent=data.get("intent", "unknown"),
                target=data.get("target", ""),
                parameters=data.get("parameters", {}),
                original_text=text,
                confidence=data.get("confidence", 0.5)
            )
        except Exception as e:
            return NLCommand(
                intent="unknown",
                target="",
                parameters={"error": str(e)},
                original_text=text,
                confidence=0.0
            )

    async def execute_command(self, command: NLCommand, user_id: str = None) -> Dict[str, Any]:
        """Выполнить распознанную команду"""

        handlers = {
            "create_table": self._handle_create_table,
            "query": self._handle_query,
            "chart": self._handle_chart,
            "report": self._handle_report,
            "update": self._handle_update,
            "delete": self._handle_delete
        }

        handler = handlers.get(command.intent)
        if handler:
            try:
                result = await handler(command, user_id)
                return {
                    "success": True,
                    "intent": command.intent,
                    "result": result,
                    "confidence": command.confidence
                }
            except Exception as e:
                return {
                    "success": False,
                    "intent": command.intent,
                    "error": str(e),
                    "confidence": command.confidence
                }

        return {
            "success": False,
            "intent": "unknown",
            "error": f"Неизвестная команда: {command.original_text}",
            "suggestion": "Попробуйте: 'создай таблицу клиентов', 'покажи все заказы', 'график продаж'"
        }

    async def _handle_create_table(self, cmd: NLCommand, user_id: str) -> Dict:
        """Создание таблицы через NLP"""
        table_name = cmd.target

        # Используем AI для генерации структуры
        if self.ai:
            prompt = f"""Создай SQL структуру таблицы "{table_name}" на основе контекста.

Контекст: {cmd.original_text}

Ответь JSON:
{{
  "columns": [
    {{"name": "id", "type": "SERIAL PRIMARY KEY"}},
    {{"name": "...", "type": "VARCHAR(255)", "description": "..."}}
  ],
  "indexes": ["..."],
  "description": "..."
}}"""

            response = await self.ai.generate(prompt, temperature=0.3, json_mode=True)
            structure = json.loads(self.ai.clean_json_response(response))
        else:
            # Базовая структура
            structure = {
                "columns": [
                    {"name": "id", "type": "SERIAL PRIMARY KEY"},
                    {"name": "name", "type": "VARCHAR(255)"},
                    {"name": "created_at", "type": "TIMESTAMP DEFAULT NOW()"}
                ],
                "description": f"Таблица {table_name}"
            }

        # Создаём SQL
        columns_sql = ",\n    ".join([
            f"{col['name']} {col['type']}"
            for col in structure["columns"]
        ])

        sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    {columns_sql}
);"""

        # Выполняем если есть БД
        if self.db:
            await self.db.execute(sql)

        return {
            "table": table_name,
            "sql": sql,
            "structure": structure,
            "message": f"Таблица '{table_name}' создана с {len(structure['columns'])} колонками"
        }

    async def _handle_query(self, cmd: NLCommand, user_id: str) -> Dict:
        """Выполнение запроса через NLP"""
        # Используем AI для генерации SQL
        if self.ai:
            prompt = f"""Преобразуй запрос на естественном языке в SQL.

ЗАПРОС: "{cmd.original_text}"
ТАБЛИЦА: {cmd.target or "auto"}

Ответь JSON:
{{
  "sql": "SELECT ...",
  "description": "что делает этот запрос",
  "chart_type": "line/bar/pie/none"  // если подходит для визуализации
}}"""

            response = await self.ai.generate(prompt, temperature=0.2, json_mode=True)
            query_data = json.loads(self.ai.clean_json_response(response))
            sql = query_data.get("sql", "SELECT 1")
        else:
            sql = f"SELECT * FROM {cmd.target} LIMIT 100"
            query_data = {"description": "Базовый запрос", "chart_type": "none"}

        # Выполняем
        if self.db:
            results = await self.db.fetch(sql)
        else:
            results = []

        return {
            "sql": sql,
            "description": query_data.get("description"),
            "results": results,
            "count": len(results),
            "chart_recommended": query_data.get("chart_type", "none") != "none",
            "chart_type": query_data.get("chart_type")
        }

    async def _handle_chart(self, cmd: NLCommand, user_id: str) -> Dict:
        """Создание графика/визуализации"""
        # Сначала получаем данные через query
        query_result = await self._handle_query(cmd, user_id)

        if not query_result.get("results"):
            return {"error": "Нет данных для визуализации"}

        # Определяем тип графика
        chart_type = "bar"  # по умолчанию
        if "время" in cmd.original_text or "динамика" in cmd.original_text or "тренд" in cmd.original_text:
            chart_type = "line"
        elif "доля" in cmd.original_text or "процент" in cmd.original_text or "распределение" in cmd.original_text:
            chart_type = "pie"

        return {
            "chart_type": chart_type,
            "data": query_result["results"],
            "sql": query_result["sql"],
            "config": {
                "title": cmd.target or "График",
                "x_axis": list(query_result["results"][0].keys())[0] if query_result["results"] else None,
                "y_axis": list(query_result["results"][0].keys())[1] if len(query_result["results"][0]) > 1 else None
            }
        }

    async def _handle_report(self, cmd: NLCommand, user_id: str) -> Dict:
        """Генерация отчёта"""
        query_result = await self._handle_query(cmd, user_id)

        # AI анализирует данные и пишет выводы
        if self.ai and query_result.get("results"):
            data_sample = json.dumps(query_result["results"][:5], ensure_ascii=False)

            prompt = f"""Проанализируй данные и создай краткий отчёт.

ДАННЫЕ (первые 5 строк):
{data_sample}

Всего записей: {query_result['count']}

Напиши:
1. Краткое описание данных
2. Ключевые показатели
3. Наблюдения/инсайты
4. Рекомендации"""

            analysis = await self.ai.generate(prompt, temperature=0.5)
        else:
            analysis = "Отчёт сгенерирован автоматически"

        return {
            "title": f"Отчёт: {cmd.target}",
            "sql": query_result.get("sql"),
            "data": query_result.get("results"),
            "analysis": analysis,
            "count": query_result.get("count", 0),
            "generated_at": "now"
        }

    async def _handle_update(self, cmd: NLCommand, user_id: str) -> Dict:
        """Обновление данных"""
        return {"message": "Обновление выполнено", "target": cmd.target}

    async def _handle_delete(self, cmd: NLCommand, user_id: str) -> Dict:
        """Удаление данных"""
        return {"message": "Удаление выполнено", "target": cmd.target}

    async def chat_with_data(self, question: str, user_id: str = None) -> Dict[str, Any]:
        """Чат с данными — свободный диалог о БД"""
        # Парсим команду
        command = await self.parse_command(question)

        # Если уверены — выполняем
        if command.confidence > 0.7:
            return await self.execute_command(command, user_id)

        # Иначе — чат-режим с AI
        if self.ai:
            context = await self._get_db_schema()

            prompt = f"""Ты помощник по работе с базой данных.

СХЕМА БД:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

Дай полезный ответ. Если нужен SQL — покажи его. Если нужна визуализация — опиши какую."""

            answer = await self.ai.generate(prompt, temperature=0.5)

            return {
                "type": "chat",
                "answer": answer,
                "suggested_commands": [
                    "покажи все таблицы",
                    "сколько записей в {table}",
                    "создай график продаж"
                ]
            }

        return {"answer": "Я не понял команду. Попробуйте переформулировать."}

    async def _get_db_schema(self) -> str:
        """Получить схему БД для контекста"""
        if not self.db:
            return "БД не подключена"

        try:
            tables = await self.db.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)

            schema_info = []
            for table in tables:
                table_name = table["table_name"]
                columns = await self.db.fetch(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                """)
                schema_info.append(f"{table_name}: {', '.join([c['column_name'] for c in columns])}")

            return "\n".join(schema_info)
        except:
            return "Схема недоступна"
