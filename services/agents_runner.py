# services/agents_runner.py
# -*- coding: utf-8 -*-
import os
import asyncio
import logging
from odoo import tools

_logger = logging.getLogger(__name__)

class AgentsRunError(Exception):
    pass

def _import_agents():
    try:
        from agents import (
            FileSearchTool,
            Agent,
            ModelSettings,
            Runner,
            RunConfig,
            trace,
            SQLiteSession,
        )
        from openai.types.shared.reasoning import Reasoning
        return FileSearchTool, Agent, ModelSettings, Runner, RunConfig, trace, SQLiteSession, Reasoning
    except Exception as e:
        raise AgentsRunError(
            'Agents SDK no instalado. Ejecuta:\n'
            'pip install "openai-agents @ git+https://github.com/openai/openai-agents-python.git"'
        ) from e

def _supports_reasoning(model: str) -> bool:
    m = (model or "").lower()
    # Los razonadores oficiales son o4, o4-mini y familia 4.1
    return m.startswith("o4") or m.startswith("gpt-4.1")

def _build_agent(env, model=None, with_tools=True):
    FileSearchTool, Agent, ModelSettings, Runner, RunConfig, trace, SQLiteSession, Reasoning = _import_agents()
    ICP = env['ir.config_parameter'].sudo()

    instructions = (
        ICP.get_param('openai_chat.agent_instructions')
        or ICP.get_param('openai_chat.system_prompt')
        or 'Eres un asistente experto en Odoo 17 Community y sus módulos.'
    )
    model = (
        model
        or ICP.get_param('openai_chat.agent_model')
        or ICP.get_param('openai_chat.model')
        or 'o4-mini'
    )
    vec_ids_str = ICP.get_param('openai_chat.agent_vector_store_ids') or ''
    vec_ids = [v.strip() for v in vec_ids_str.split(',') if v.strip()]

    tools_list = []
    if with_tools and vec_ids:
        tools_list.append(FileSearchTool(vector_store_ids=vec_ids))

    # reasoning solo si el modelo lo soporta
    ms_kwargs = dict(store=True)
    if _supports_reasoning(model):
        try:
            from openai.types.shared.reasoning import Reasoning
            ms_kwargs['reasoning'] = Reasoning(effort="high", summary="auto")
        except Exception:
            pass

    agent = Agent(
        name="Odoo Support Agent",
        instructions=instructions,
        model=model,
        tools=tools_list,
        model_settings=ModelSettings(**ms_kwargs),
    )
    return agent, bool(vec_ids)

def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.stop()
        except Exception:
            pass
        loop.close()
        asyncio.set_event_loop(None)

def run_agent_for_channel(env, channel, user_prompt):
    FileSearchTool, Agent, ModelSettings, Runner, RunConfig, trace, SQLiteSession, Reasoning = _import_agents()
    ICP = env['ir.config_parameter'].sudo()

    api_key = ICP.get_param('openai_chat.api_key')
    base_url = (ICP.get_param('openai_chat.base_url') or 'https://api.openai.com/v1').rstrip('/')
    timeout = int(ICP.get_param('openai_chat.timeout') or 60)
    if not api_key:
        raise AgentsRunError('Falta API Key para Agents SDK')

    # Variables de entorno (el SDK suele leer estas)
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = base_url
    os.environ['OPENAI_API_BASE'] = base_url  # por compatibilidad

    # Sesión SQLite por canal
    filestore_dir = tools.config.filestore(env.cr.dbname)
    agents_dir = os.path.join(filestore_dir, 'openai_agents')
    os.makedirs(agents_dir, exist_ok=True)
    session_path = os.path.join(agents_dir, f'channel_{channel.id}.sqlite3')
    session = SQLiteSession(session_path)

    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
    ]

    def _extract_text(result):
        text = None
        if hasattr(result, 'final_output_as'):
            try:
                text = result.final_output_as(str)
            except Exception:
                text = None
        if not text:
            text = getattr(result, 'final_output', None) or getattr(result, 'output', None)
        return str(text) if text is not None else ''

    async def _call(agent):
        with trace("Odoo Agents Run"):
            try:
                return await Runner.run(
                    agent,
                    input=input_items,
                    session=session,
                    run_config=RunConfig(trace_metadata={
                        "__trace_source__": "odoo",
                        "channel_id": str(channel.id),
                    }),
                    request_timeout=timeout,
                )
            except TypeError:
                # versiones que no aceptan request_timeout
                return await Runner.run(
                    agent,
                    input=input_items,
                    session=session,
                    run_config=RunConfig(trace_metadata={
                        "__trace_source__": "odoo",
                        "channel_id": str(channel.id),
                    }),
                )

    # Estrategia en cascada:
    # 1) Modelo configurado con tools (si hay)
    # 2) Modelo configurado sin tools
    # 3) Modelo fallback (o4-mini) con tools
    # 4) Modelo fallback sin tools
    errors = []

    for step, (model, with_tools) in enumerate([
        (None, True),
        (None, False),
        ('o4-mini', True),
        ('o4-mini', False),
    ], start=1):
        try:
            agent, had_vec = _build_agent(env, model=model, with_tools=with_tools)
            _logger.info("Agents step %s -> model=%s, tools=%s, channel=%s", step, agent.model, with_tools and had_vec, channel.id)
            result = _run_async(_call(agent))
            text = _extract_text(result)
            if text:
                return text
            errors.append(f"step {step}: respuesta vacía")
        except Exception as e:
            msg = f"step {step} error: {type(e).__name__}: {e}"
            errors.append(msg)
            _logger.warning("Agents: %s", msg, exc_info=True)

    # Si todas fallan, relanza para que _generate_ai_reply haga fallback a Chat
    raise AgentsRunError("; ".join(errors))
