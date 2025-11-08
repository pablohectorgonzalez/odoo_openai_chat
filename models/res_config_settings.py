# -*- coding: utf-8 -*-
import logging
from odoo import api, fields, models

_logger = logging.getLogger(__name__)

class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    openai_enabled = fields.Boolean(string='Habilitar OpenAI Chat', config_parameter='openai_chat.enabled')
    openai_api_key = fields.Char(string='OpenAI API Key', config_parameter='openai_chat.api_key')
    openai_organization = fields.Char(string='OpenAI Organization', config_parameter='openai_chat.organization')
    openai_base_url = fields.Char(string='OpenAI Base URL', default='https://api.openai.com/v1', config_parameter='openai_chat.base_url')
    openai_model = fields.Char(string='Modelo', default='gpt-4o-mini', config_parameter='openai_chat.model')
    openai_temperature = fields.Float(string='Temperature', default=0.2, config_parameter='openai_chat.temperature')
    openai_context_count = fields.Integer(string='Contexto (mensajes)', default=10, config_parameter='openai_chat.context_count')
    openai_timeout = fields.Integer(string='Timeout (s)', default=60, config_parameter='openai_chat.timeout')
    openai_system_prompt_char = fields.Char(string='System prompt', config_parameter='openai_chat.system_prompt', help='Prompt del sistema para OpenAI')

    openai_enabled = fields.Boolean(
        string='Habilitar OpenAI en Discuss',
        config_parameter='openai_chat.enabled',
    )
    openai_api_key = fields.Char(
        string='OpenAI API Key',
        help='Se almacenará en parámetros del sistema',
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
        help='Cambiar si usas un proxy o Azure (compatible con API de OpenAI)',
        config_parameter='openai_chat.base_url'
    )
    openai_model = fields.Char(
        string='Modelo por defecto',
        default='gpt-4o-mini',
        help='Ejemplo: gpt-4o-mini, gpt-4.1-mini',
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
        help='Cantidad de mensajes recientes del canal a enviar como contexto',
        config_parameter='openai_chat.context_count'
    )
    openai_system_prompt_char = fields.Char(
        string='Prompt de sistema',
        default='Eres un asistente útil para usuarios de Odoo. Responde de forma breve y clara.',
        config_parameter='openai_chat.system_prompt'
    )
    openai_timeout = fields.Integer(
        string='Timeout (segundos)',
        default=60,
        config_parameter='openai_chat.timeout'
    )

    @api.onchange('openai_base_url')
    def _onchange_openai_base_url(self):
        if self.openai_base_url:
            self.openai_base_url = self.openai_base_url.rstrip('/')
