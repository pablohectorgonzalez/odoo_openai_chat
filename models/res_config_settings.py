import logging
import json
import requests
from odoo import api, fields, models, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    # Núcleo
    openai_enabled = fields.Boolean(
        string='Habilitar OpenAI en Discuss',
        config_parameter='openai_chat.enabled',
    )
    openai_api_key = fields.Char(
        string='OpenAI API Key',
        help='Se almacenará en Parámetros del sistema',
        config_parameter='openai_chat.api_key',
        groups='base.group_system'
    )
    openai_organization = fields.Char(
        string='OpenAI Organization (opcional)',
        config_parameter='openai_chat.organization',
        groups='base.group_system'
    )
    openai_base_url = fields.Char(
        string='OpenAI Base URL',
        default='https://api.openai.com/v1',
        help='Cámbialo si usas un proxy o Azure (compatible con API de OpenAI)',
        config_parameter='openai_chat.base_url'
    )
    openai_model = fields.Char(
        string='Modelo por defecto',
        default='gpt-4o-mini',
        help='Ej.: gpt-4o-mini, gpt-4.1-mini',
        config_parameter='openai_chat.model'
    )
    openai_temperature = fields.Float(
        string='Temperatura',
        default=0.2,
        config_parameter='openai_chat.temperature'
    )
    openai_context_count = fields.Integer(
        string='Mensajes de contexto',
        default=10,
        help='Cuántos mensajes recientes enviar como contexto (para Chat Completions)',
        config_parameter='openai_chat.context_count'
    )
    openai_system_prompt = fields.Char(
        string='Prompt de sistema',
        default='Eres un asistente útil para usuarios de Odoo. Responde de forma breve y clara.',
        config_parameter='openai_chat.system_prompt'
    )
    openai_timeout = fields.Integer(
        string='Timeout (segundos)',
        default=60,
        config_parameter='openai_chat.timeout'
    )

    # Assistants v2
    openai_assistant_id = fields.Char(
        string='Assistant ID (asst_...)',
        help='ID del Assistant (Assistants API v2)',
        config_parameter='openai_chat.assistant_id'
    )

    # Agents SDK (Runner)
    openai_agent_mode = fields.Selection(
        selection=[
            ('chat', 'Chat Completions'),
            ('assistants', 'Assistants API v2'),
            ('agents', 'Agents SDK (Runner)'),
        ],
        string='Modo de agente',
        default='chat',
        config_parameter='openai_chat.agent_mode',
        help='Selecciona la integración preferida. Si falla, hará fallback automático.'
    )
    openai_agent_instructions = fields.Char(
        string='Instrucciones del Agente',
        help='Instrucciones base para el Agent Runner (si usas Agent Builder/Runner)',
        config_parameter='openai_chat.agent_instructions'
    )

    # Añade a models/res_config_settings.py (dentro de ResConfigSettings)
    openai_agent_model = fields.Char(
        string='Modelo (Agents)',
        help='Modelo a usar solo para el modo Agents. Ej.: gpt-5, gpt-4.1, gpt-4o-mini',
        default='gpt-4o-mini',
        config_parameter='openai_chat.agent_model',
    )
    openai_agent_vector_store_ids = fields.Char(
        string='Vector Store IDs',
        help='IDs separados por coma (vs_...) para FileSearchTool',
        config_parameter='openai_chat.agent_vector_store_ids',
    )

    @api.onchange('openai_base_url')
    def _onchange_openai_base_url(self):
        if self.openai_base_url:
            self.openai_base_url = self.openai_base_url.rstrip('/')

    def action_test_openai(self):
        self.ensure_one()
        from ..services.diagnostics import run_deep_diagnostics, format_report
        rep = run_deep_diagnostics(self.env, prompt='Di "pong".', timeout=20)
        msg = format_report(rep)
        notif_type = 'success' if rep.get('summary', {}).get('ok') else 'warning'
        if rep.get('summary', {}).get('errors'):
            notif_type = 'danger'
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'OpenAI Diagnostics',
                'message': msg[:1000],
                'type': notif_type,
                'sticky': False,
            }
        }

    @api.onchange('openai_base_url')
    def _onchange_openai_base_url(self):
        if self.openai_base_url:
            self.openai_base_url = self.openai_base_url.rstrip('/')


    
