# services/agents_runner.py
# -*- coding: utf-8 -*-
import os
import asyncio
import logging

from odoo import tools

_logger = logging.getLogger(__name__)

def _import_agents():
    """
    Importa el SDK de Agents desde rutas conocidas.
    pip: pip install "openai-agents @ git+https://github.com/openai/openai-agents-python.git"
    """
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
        raise RuntimeError(
            "Agents SDK no instalado. Ejecuta:\n"
            'pip install "openai-agents @ git+https://github.com/openai/openai-agents-python.git"'
        ) from e

def _build_agent(env):
    """
    Construye el agente usando parámetros de Odoo:
    - Instrucciones: openai_chat.agent_instructions o system_prompt
    - Modelo: openai_chat.agent_model (fallback a openai_chat.model o gpt-4o-mini)
    - Vector Stores: openai_chat.agent_vector_store_ids (coma-separado)
    """
    FileSearchTool, Agent, ModelSettings, Runner, RunConfig, trace, SQLiteSession, Reasoning = _import_agents()
    ICP = env['ir.config_parameter'].sudo()

    instructions = ICP.get_param('openai_chat.agent_instructions') or \
                   ICP.get_param('openai_chat.system_prompt') or \
                   'Eres un asistente útil para Odoo 17.'
    model = ICP.get_param('openai_chat.agent_model') or \
            ICP.get_param('openai_chat.model') or \
            'gpt-4o-mini'
    vec_ids_str = ICP.get_param('openai_chat.agent_vector_store_ids') or ''
    vec_ids = [v.strip() for v in vec_ids_str.split(',') if v.strip()]

    tools_list = []
    if vec_ids:
        tools_list.append(FileSearchTool(vector_store_ids=vec_ids))

    agent = Agent(
        name="Odoo Support Agent",
        instructions=instructions,
        model=model,
        tools=tools_list,
        model_settings=ModelSettings(
            store=True,
            reasoning=Reasoning(
                effort="high",
                summary="auto",
            ),
        ),
    )
    return agent

def run_agent_for_channel(env, channel, user_prompt):
    """
    Ejecuta el agente (Agents SDK) para un canal de Discuss.
    - Memoria por canal con SQLiteSession en filestore: .../filestore/DB/openai_agents/channel_{id}.sqlite3
    - Entrada como input_text (TResponseInputItem)
    - Devuelve el texto del final_output.
    """
    FileSearchTool, Agent, ModelSettings, Runner, RunConfig, trace, SQLiteSession, Reasoning = _import_agents()
    ICP = env['ir.config_parameter'].sudo()

    api_key = ICP.get_param('openai_chat.api_key')
    base_url = (ICP.get_param('openai_chat.base_url') or 'https://api.openai.com/v1').rstrip('/')
    timeout = int(ICP.get_param('openai_chat.timeout') or 60)
    if not api_key:
        raise ValueError('Falta API Key para Agents SDK')

    # Variables de entorno (algunos clientes las leen implícitamente)
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = base_url

    # Sesión SQLite por canal
    filestore_dir = tools.config.filestore(env.cr.dbname)
    agents_dir = os.path.join(filestore_dir, 'openai_agents')
    os.makedirs(agents_dir, exist_ok=True)
    session_path = os.path.join(agents_dir, f'channel_{channel.id}.sqlite3')
    session = SQLiteSession(session_path)

    agent = _build_agent(env)

    # Entrada del usuario (TResponseInputItem[])
    input_items = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt}
            ]
        }
    ]

    async def _run():
        with trace("Odoo Agents Run"):
            result = await Runner.run(
                agent,
                input=input_items,
                session=session,
                run_config=RunConfig(trace_metadata={
                    "__trace_source__": "odoo",
                    "channel_id": str(channel.id),
                }),
            )
            # final_output_as(str) si está disponible (Agent Builder)
            text = None
            if hasattr(result, 'final_output_as'):
                try:
                    text = result.final_output_as(str)
                except Exception:
                    text = None
            if not text:
                text = getattr(result, 'final_output', None) or getattr(result, 'output', None)
            return str(text) if text is not None else ''

    return asyncio.run(_run())
