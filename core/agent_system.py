import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class AgentTask:
    id: str
    type: str
    params: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: str = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str = None

@dataclass
class Agent:
    id: str
    name: str
    description: str
    capabilities: List[str]
    system_prompt: str
    tools: List[str]
    memory: List[Dict] = field(default_factory=list)
    max_memory: int = 10

class AgentSystem:
    """Система ИИ-агентов с интеграцией Flowise и Activepieces"""

    def __init__(self, ai_manager=None, rag_engine=None, db=None):
        self.ai = ai_manager
        self.rag = rag_engine
        self.db = db

        # URL внешних сервисов
        self.flowise_url = os.getenv("FLOWISE_URL", "http://localhost:3000")
        self.flowise_api_key = os.getenv("FLOWISE_API_KEY")

        self.activepieces_url = os.getenv("ACTIVEPIECES_URL", "http://localhost:4200")
        self.activepieces_api_key = os.getenv("ACTIVEPIECES_API_KEY")

        # Агенты
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, AgentTask] = {}

        # Инструменты
        self.tools: Dict[str, Callable] = {
            "search_web": self._tool_search_web,
            "search_docs": self._tool_search_docs,
            "generate_code": self._tool_generate_code,
            "generate_image": self._tool_generate_image,
            "send_notification": self._tool_send_notification,
            "save_to_db": self._tool_save_to_db,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "execute_command": self._tool_execute_command,
            "api_request": self._tool_api_request,
            "schedule_task": self._tool_schedule_task
        }

        # Инициализация стандартных агентов
        self._init_default_agents()

    def _init_default_agents(self):
        """Создать стандартных агентов"""
        default_agents = [
            Agent(
                id="developer",
                name="Разработчик",
                description="Пишет код, создаёт проекты, ревьюит",
                capabilities=["coding", "architecture", "debugging"],
                system_prompt="""Ты опытный разработчик ПО. Твои задачи:
- Писать чистый, документированный код
- Создавать архитектуру приложений
- Делать code review
- Исправлять баги
- Оптимизировать производительность

Отвечай структурированно, с примерами кода.""",
                tools=["generate_code", "read_file", "write_file", "execute_command", "search_web"]
            ),
            Agent(
                id="analyst",
                name="Аналитик",
                description="Анализирует данные, создаёт отчёты",
                capabilities=["analysis", "reporting", "visualization"],
                system_prompt="""Ты data analyst. Твои задачи:
- Анализировать данные
- Создавать SQL запросы
- Строить графики и дашборды
- Писать отчёты
- Находить инсайты в данных

Используй конкретные цифры и факты.""",
                tools=["search_docs", "save_to_db", "api_request", "generate_code"]
            ),
            Agent(
                id="support",
                name="Поддержка",
                description="Отвечает на вопросы, решает проблемы",
                capabilities=["support", "troubleshooting", "communication"],
                system_prompt="""Ты специалист поддержки. Твои задачи:
- Отвечать на вопросы пользователей
- Решать технические проблемы
- Эскалировать сложные кейсы
- Вести базу знаний

Будь вежливым, терпеливым и точным.""",
                tools=["search_docs", "send_notification", "save_to_db", "schedule_task"]
            ),
            Agent(
                id="manager",
                name="Менеджер проектов",
                description="Планирует, ставит задачи, контролирует",
                capabilities=["planning", "management", "coordination"],
                system_prompt="""Ты project manager. Твои задачи:
- Ставить задачи команде
- Планировать сроки
- Контролировать выполнение
- Проводить встречи
- Писать ТЗ

Будь организованным и чётким.""",
                tools=["send_notification", "schedule_task", "save_to_db", "api_request"]
            ),
            Agent(
                id="creative",
                name="Креативщик",
                description="Генерирует идеи, контент, дизайн",
                capabilities=["ideation", "content", "design"],
                system_prompt="""Ты креативный директор. Твои задачи:
- Генерировать идеи
- Писать тексты
- Создавать концепции
- Давать имена
- Разрабатывать креативы

Будь креативным, но практичным.""",
                tools=["generate_image", "generate_code", "search_web", "save_to_db"]
            )
        ]

        for agent in default_agents:
            self.agents[agent.id] = agent

    async def create_agent(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        system_prompt: str,
        tools: List[str],
        custom_params: Dict = None
    ) -> Agent:
        """Создать нового агента"""
        agent = Agent(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            capabilities=capabilities,
            system_prompt=system_prompt,
            tools=[t for t in tools if t in self.tools],
            max_memory=custom_params.get("max_memory", 10) if custom_params else 10
        )

        self.agents[agent.id] = agent

        # Сохраняем в БД если есть
        if self.db:
            await self.db.save_agent(agent)

        return agent

    async def run_agent(
        self,
        agent_id: str,
        user_input: str,
        context: Dict = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Запустить агента с задачей"""

        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}

        # Создаём задачу
        task = AgentTask(
            id=str(uuid.uuid4()),
            type="agent_run",
            params={"agent_id": agent_id, "input": user_input, "context": context}
        )
        self.tasks[task.id] = task

        try:
            task.status = "running"

            # Формируем промпт
            memory_context = self._format_memory(agent.memory)

            full_prompt = f"""{agent.system_prompt}

КОНТЕКСТ ИЗ ПАМЯТИ:
{memory_context}

ТЕКУЩИЙ ЗАПРОС: {user_input}

ДОСТУПНЫЕ ИНСТРУМЕНТЫ: {', '.join(agent.tools)}

Если нужно использовать инструмент, ответь в формате:
ACTION: tool_name
PARAMS: {{"param1": "value1"}}

Иначе просто ответь на вопрос."""

            # Генерируем ответ
            if self.ai:
                response = await self.ai.generate(full_prompt, temperature=0.7)
            else:
                response = "AI не настроен"

            # Проверяем, хочет ли агент использовать инструмент
            if "ACTION:" in response:
                tool_result = await self._execute_tool_from_response(response, agent)

                # Добавляем результат в контекст и перегенерируем
                full_prompt += f"\n\nРЕЗУЛЬТАТ ИНСТРУМЕНТА: {json.dumps(tool_result, ensure_ascii=False)}"
                final_response = await self.ai.generate(full_prompt, temperature=0.7)
            else:
                final_response = response
                tool_result = None

            # Обновляем память
            agent.memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.utcnow().isoformat()
            })
            agent.memory.append({
                "role": "assistant",
                "content": final_response,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Ограничиваем память
            if len(agent.memory) > agent.max_memory * 2:
                agent.memory = agent.memory[-agent.max_memory * 2:]

            task.status = "completed"
            task.result = {
                "response": final_response,
                "tool_used": tool_result is not None,
                "tool_result": tool_result
            }
            task.completed_at = datetime.utcnow().isoformat()

            return {
                "task_id": task.id,
                "agent": agent.name,
                "response": final_response,
                "tool_used": tool_result is not None,
                "memory_items": len(agent.memory) // 2
            }

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            return {"error": str(e), "task_id": task.id}

    async def _execute_tool_from_response(self, response: str, agent: Agent) -> Dict:
        """Извлечь и выполнить инструмент из ответа агента"""
        try:
            lines = response.split('\n')
            tool_name = None
            params = {}

            for line in lines:
                if line.startswith("ACTION:"):
                    tool_name = line.replace("ACTION:", "").strip()
                elif line.startswith("PARAMS:"):
                    params_str = line.replace("PARAMS:", "").strip()
                    params = json.loads(params_str)

            if tool_name and tool_name in agent.tools:
                tool_func = self.tools.get(tool_name)
                if tool_func:
                    return await tool_func(**params)

            return {"error": f"Tool {tool_name} not found or not allowed"}

        except Exception as e:
            return {"error": str(e)}

    def _format_memory(self, memory: List[Dict]) -> str:
        """Форматировать память агента"""
        if not memory:
            return "Нет предыдущего контекста"

        formatted = []
        for item in memory[-10:]:  # Последние 10 сообщений
            role = "Пользователь" if item["role"] == "user" else "Ассистент"
            formatted.append(f"{role}: {item['content'][:200]}")

        return "\n".join(formatted)

    # === TOOLS ===

    async def _tool_search_web(self, query: str, **kwargs) -> Dict:
        """Поиск в интернете (через Serper или similar)"""
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {"error": "Web search not configured"}

        async with aiohttp.ClientSession() as session:
            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
            payload = {"q": query, "num": kwargs.get("num", 5)}

            async with session.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload
            ) as resp:
                data = await resp.json()
                return {
                    "results": data.get("organic", []),
                    "query": query
                }

    async def _tool_search_docs(self, query: str, collection: str = "default", **kwargs) -> Dict:
        """Поиск в документации (RAG)"""
        if not self.rag:
            return {"error": "RAG not available"}

        results = await self.rag.search(collection, query, top_k=kwargs.get("top_k", 5))
        return {
            "documents": results,
            "query": query
        }

    async def _tool_generate_code(self, description: str, language: str = "python", **kwargs) -> Dict:
        """Генерация кода"""
        if not self.ai:
            return {"error": "AI not available"}

        prompt = f"Напиши {language} код для: {description}"
        code = await self.ai.generate(prompt, temperature=0.3)

        return {
            "code": code,
            "language": language,
            "description": description
        }

    async def _tool_generate_image(self, prompt: str, **kwargs) -> Dict:
        """Генерация изображения"""
        # Делегируем в MediaProcessor
        return {"status": "delegated_to_media_processor", "prompt": prompt}

    async def _tool_send_notification(self, message: str, channel: str = "telegram", **kwargs) -> Dict:
        """Отправка уведомления"""
        if channel == "telegram":
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = kwargs.get("chat_id")

            if not bot_token or not chat_id:
                return {"error": "Telegram not configured"}

            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }

                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    return {
                        "sent": data.get("ok"),
                        "message_id": data.get("result", {}).get("message_id")
                    }

        return {"error": f"Channel {channel} not supported"}

    async def _tool_save_to_db(self, data: Dict, table: str = "logs", **kwargs) -> Dict:
        """Сохранение в БД"""
        if not self.db:
            return {"error": "Database not available"}

        try:
            await self.db.insert(table, data)
            return {"saved": True, "table": table}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_read_file(self, path: str, **kwargs) -> Dict:
        """Чтение файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content, "path": path}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_write_file(self, path: str, content: str, **kwargs) -> Dict:
        """Запись файла"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"written": True, "path": path, "size": len(content)}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_execute_command(self, command: str, **kwargs) -> Dict:
        """Выполнение команды (ограниченное)"""
        # Безопасность: только разрешённые команды
        allowed_prefixes = ["ls", "cat", "grep", "find", "echo", "pwd"]

        if not any(command.startswith(prefix) for prefix in allowed_prefixes):
            return {"error": "Command not allowed for security reasons"}

        try:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_api_request(self, url: str, method: str = "GET", **kwargs) -> Dict:
        """HTTP запрос"""
        async with aiohttp.ClientSession() as session:
            try:
                if method == "GET":
                    async with session.get(url, params=kwargs.get("params"), headers=kwargs.get("headers")) as resp:
                        return {
                            "status": resp.status,
                            "data": await resp.json() if resp.content_type == 'application/json' else await resp.text()
                        }
                elif method == "POST":
                    async with session.post(url, json=kwargs.get("body"), headers=kwargs.get("headers")) as resp:
                        return {
                            "status": resp.status,
                            "data": await resp.json() if resp.content_type == 'application/json' else await resp.text()
                        }
            except Exception as e:
                return {"error": str(e)}

    async def _tool_schedule_task(self, task_type: str, when: str, params: Dict, **kwargs) -> Dict:
        """Планирование задачи"""
        # Интеграция с Activepieces или внутренний планировщик
        task_id = str(uuid.uuid4())

        # Сохраняем в БД для обработки планировщиком
        if self.db:
            await self.db.insert("scheduled_tasks", {
                "id": task_id,
                "type": task_type,
                "when": when,
                "params": json.dumps(params),
                "status": "scheduled"
            })

        return {
            "scheduled": True,
            "task_id": task_id,
            "when": when
        }

    # === Flowise Integration ===

    async def create_flowise_agent(self, name: str, flow_config: Dict) -> Dict:
        """Создать агента через Flowise API"""
        if not self.flowise_api_key:
            return {"error": "Flowise not configured"}

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.flowise_api_key}"}

            # Создаём чатфлоу
            async with session.post(
                f"{self.flowise_url}/api/v1/chatflows",
                headers=headers,
                json={"name": name, "flowData": json.dumps(flow_config)}
            ) as resp:
                if resp.status in [200, 201]:
                    data = await resp.json()
                    return {
                        "success": True,
                        "chatflow_id": data.get("id"),
                        "name": name
                    }
                else:
                    return {"error": await resp.text()}

    async def run_flowise_agent(self, chatflow_id: str, question: str) -> Dict:
        """Запустить Flowise агента"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.flowise_url}/api/v1/prediction/{chatflow_id}",
                json={"question": question}
            ) as resp:
                return await resp.json()

    # === Activepieces Integration ===

    async def create_activepieces_flow(self, name: str, steps: List[Dict]) -> Dict:
        """Создать автоматизацию в Activepieces"""
        if not self.activepieces_api_key:
            return {"error": "Activepieces not configured"}

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.activepieces_api_key}"}

            async with session.post(
                f"{self.activepieces_url}/api/v1/flows",
                headers=headers,
                json={"displayName": name, "steps": steps}
            ) as resp:
                if resp.status in [200, 201]:
                    data = await resp.json()
                    return {
                        "success": True,
                        "flow_id": data.get("id"),
                        "name": name
                    }
                else:
                    return {"error": await resp.text()}

    def get_agents(self) -> List[Dict]:
        """Получить список всех агентов"""
        return [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "capabilities": a.capabilities,
                "tools": a.tools,
                "memory_items": len(a.memory) // 2
            }
            for a in self.agents.values()
        ]

    def get_tasks(self, status: str = None) -> List[Dict]:
        """Получить задачи"""
        tasks = self.tasks.values()
        if status:
            tasks = [t for t in tasks if t.status == status]

        return [
            {
                "id": t.id,
                "type": t.type,
                "status": t.status,
                "created_at": t.created_at,
                "completed_at": t.completed_at
            }
            for t in tasks
        ]
