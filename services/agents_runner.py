# -*- coding: utf-8 -*-
import os
import asyncio
import logging
import uuid

_logger = logging.getLogger(__name__)

def run_agent_for_channel(env, channel, user_prompt):
    """
    Ejecuta el Agents SDK (Runner) para un canal de Discuss.
    - Requiere pip: openai (>=1.x) y openai-agents-python (módulo 'agents')
    - Usa conversation_id por canal (channel.openai_agent_conversation_id)
    - Lee instrucciones del agente desde openai_chat.agent_instructions (si no, usa system_prompt)
    """
    try:
        # Imports del SDK
        from agents import Agent, Runner  # from openai-agents-python
        from openai import AsyncOpenAI
    except Exception as e:
        raise RuntimeError("No está instalado el SDK de Agents. Instala 'openai' y 'openai-agents-python'") from e

    ICP = env['ir.config_parameter'].sudo()
    api_key = ICP.get_param('openai_chat.api_key')
    base_url = (ICP.get_param('openai_chat.base_url') or 'https://api.openai.com/v1').rstrip('/')
    instructions = ICP.get_param('openai_chat.agent_instructions') or ICP.get_param('openai_chat.system_prompt') or 'Eres un asistente útil para usuarios de Odoo.'
    timeout = int(ICP.get_param('openai_chat.timeout') or 60)

    if not api_key:
        raise ValueError('Falta API Key para Agents SDK')

    # Asegura variables de entorno para clientes que las usan implícitamente
    os.environ['OPENAI_API_KEY'] = api_key
    # Nota: no todos los SDK respetan OPENAI_BASE_URL; si no se respeta, el modo agents solo funcionará contra api.openai.com
    os.environ['OPENAI_BASE_URL'] = base_url

    # Conversation ID por canal
    conv_id = channel.openai_agent_conversation_id
    if not conv_id:
        # Intenta conversación manejada por OpenAI; si no existe el endpoint, crea un UUID local.
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            async def _create_conv():
                conv = await client.conversations.create()
                return conv.id
            conv_id = asyncio.run(_create_conv())
        except Exception as e:
            _logger.warning("No se pudo crear conversation_id en OpenAI, se usará UUID local: %s", e)
            conv_id = f'odoo-{env.cr.dbname}-channel-{channel.id}-{uuid.uuid4().hex}'
        channel.sudo().write({'openai_agent_conversation_id': conv_id})
        channel.env.cr.commit()

    # Define el agente con tus instrucciones (puedes enriquecer con tools/handoffs si lo exportaste del Agent Builder)
    agent = Agent(
        name="Odoo Support Agent",
        instructions=instructions
    )

    async def _run():
        # Runner.run orquesta herramientas/handoffs si el agente las declara
        result = await Runner.run(agent, user_prompt, conversation_id=conv_id)
        # Algunas versiones exponen final_output; si no, usa result.output o similar
        return getattr(result, 'final_output', None) or getattr(result, 'output', None) or str(result)

    return asyncio.run(_run())
