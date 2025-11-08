# -*- coding: utf-8 -*-
import logging
import re
import requests

from odoo import api, fields, models, tools, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

AI_PARTNER_XMLID = 'openai_chat.partner_ai'  # si luego quieres crear por XML
AI_PARTNER_NAME = 'AI Assistant'
AI_PARTNER_EMAIL = 'assistant@openai.local'

class MailChannel(models.Model):
    _inherit = 'mail.channel'

    def message_post(self, **kwargs):
        # Guard para evitar recursión al publicar respuesta del bot
        if self.env.context.get('openai_skip'):
            return super(MailChannel, self).message_post(**kwargs)

        body_html = kwargs.get('body') or ''
        body_plain = tools.html2plaintext(body_html or '').strip()

        # Post original del usuario
        message = super(MailChannel, self).message_post(**kwargs)

        try:
            if not self._openai_is_enabled():
                return message

            # Detectar trigger: /ai en cualquier canal
            trigger, user_prompt = self._detect_ai_trigger(body_plain)

            # O conversación 1:1 con el bot (sin /ai)
            if not trigger and self._is_dm_with_ai_bot():
                # En chat 1:1 con el bot, cualquier mensaje del usuario dispara respuesta
                # exceptuando mensajes del propio bot.
                author_partner_id = kwargs.get('author_id') or self.env.user.partner_id.id
                ai_partner = self._get_or_create_ai_partner()
                if author_partner_id != ai_partner.id:
                    trigger = True
                    user_prompt = body_plain

            if not trigger:
                return message

            # Construción del contexto y llamada a OpenAI
            cfg = self._get_openai_config()
            if not cfg.get('api_key'):
                self._post_ai_error(_('Falta el API Key de OpenAI en Ajustes'), as_thread=True)
                return message

            ai_partner = self._get_or_create_ai_partner()

            # Preparar mensajes para Chat Completions
            messages_payload = self._prepare_openai_chat_messages(user_prompt=user_prompt, ai_partner=ai_partner, cfg=cfg, exclude_message_id=message.id)

            # Llamada a OpenAI Chat Completions
            reply_text = self._call_openai_chat(messages_payload=messages_payload, cfg=cfg)

            if not reply_text:
                reply_text = _('No se pudo obtener respuesta del modelo.')

            # Publicar respuesta del bot
            self.with_context(openai_skip=True).message_post(
                body=tools.plaintext2html(reply_text),
                author_id=ai_partner.id,
                message_type='comment',
                subtype_xmlid='mail.mt_comment',
            )
        except Exception as e:
            _logger.exception('Error al procesar respuesta de OpenAI: %s', e)
            self._post_ai_error(_('Error al procesar la respuesta de OpenAI: %s') % e, as_thread=True)

        return message

    # ---------------------------
    # Helpers de configuración
    # ---------------------------
    def _openai_is_enabled(self):
        icp = self.env['ir.config_parameter'].sudo()
        return tools.str2bool(icp.get_param('openai_chat.enabled', 'False'))

    def _get_openai_config(self):
        icp = self.env['ir.config_parameter'].sudo()
        return {
            'api_key': icp.get_param('openai_chat.api_key'),
            'organization': icp.get_param('openai_chat.organization') or None,
            'base_url': (icp.get_param('openai_chat.base_url') or 'https://api.openai.com/v1').rstrip('/'),
            'model': icp.get_param('openai_chat.model') or 'gpt-4o-mini',
            'temperature': float(icp.get_param('openai_chat.temperature') or 0.2),
            'context_count': int(icp.get_param('openai_chat.context_count') or 10),
            'system_prompt': icp.get_param('openai_chat.system_prompt') or 'Eres un asistente útil para usuarios de Odoo. Responde de forma breve y clara.',
            'timeout': int(icp.get_param('openai_chat.timeout') or 60),
        }

    # ---------------------------
    # Triggers y detección
    # ---------------------------
    def _detect_ai_trigger(self, body_plain):
        """
        Detecta si el mensaje inicia con /ai y retorna (trigger, prompt)
        """
        if not body_plain:
            return False, ''
        m = re.match(r'^\s*/ai\s*(.*)$', body_plain, re.S | re.I)
        if m:
            return True, (m.group(1) or '').strip()
        return False, ''

    def _is_dm_with_ai_bot(self):
        """
        Retorna True si el canal es un chat 1:1 entre el usuario y el bot
        """
        self.ensure_one()
        if self.channel_type != 'chat':
            return False
        partners = self.channel_partner_ids
        if len(partners) != 2:
            return False
        ai_partner = self._get_or_create_ai_partner()
        return ai_partner in partners

    # ---------------------------
    # OpenAI: preparación y llamada
    # ---------------------------
    def _prepare_openai_chat_messages(self, user_prompt, ai_partner, cfg, exclude_message_id=None):
        """
        Transforma el histórico del canal en mensajes para Chat Completions.
        - Incluye system prompt
        - Toma los últimos N mensajes (cfg['context_count'])
        - Excluye el mensaje recién creado (exclude_message_id)
        - Añade el user_prompt como último mensaje user
        """
        self.ensure_one()

        messages = []
        system_prompt = (cfg.get('system_prompt') or '').strip()
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        domain = [
            ('model', '=', 'mail.channel'),
            ('res_id', '=', self.id),
            ('message_type', 'in', ['comment', 'email']),
            ('body', '!=', False),
        ]
        if exclude_message_id:
            domain.append(('id', '!=', exclude_message_id))

        # Traemos los últimos N mensajes y los ordenamos ascendentemente para mantener el orden cronológico
        last_msgs = self.env['mail.message'].sudo().search(domain, order='id desc', limit=cfg['context_count'])
        for msg in reversed(last_msgs):
            role = 'assistant' if (msg.author_id and msg.author_id.id == ai_partner.id) else 'user'
            content = tools.html2plaintext(msg.body or '').strip()
            if not content:
                continue
            messages.append({'role': role, 'content': content})

        # Finalmente el mensaje actual del usuario (sin /ai)
        if user_prompt:
            messages.append({'role': 'user', 'content': user_prompt})

        return messages

    def _call_openai_chat(self, messages_payload, cfg):
        """
        Llama a OpenAI Chat Completions API con requests para evitar dependencias extra.
        """
        url = f"{cfg['base_url']}/chat/completions"
        headers = {
            'Authorization': f"Bearer {cfg['api_key']}",
            'Content-Type': 'application/json',
        }
        if cfg.get('organization'):
            headers['OpenAI-Organization'] = cfg['organization']

        payload = {
            'model': cfg['model'],
            'messages': messages_payload,
            'temperature': cfg['temperature'],
        }

        _logger.debug('OpenAI request to %s: %s', url, payload)

        resp = requests.post(url, json=payload, headers=headers, timeout=cfg['timeout'])
        if resp.status_code >= 400:
            _logger.error('OpenAI error %s: %s', resp.status_code, resp.text)
            raise UserError(_('Error %s desde OpenAI: %s') % (resp.status_code, resp.text))

        data = resp.json()
        _logger.debug('OpenAI response: %s', data)
        try:
            return data['choices'][0]['message']['content']
        except Exception:
            _logger.error('No se pudo parsear la respuesta de OpenAI: %s', data)
            return ''

    # ---------------------------
    # Utilidades
    # ---------------------------
    def _get_or_create_ai_partner(self):
        """
        Crea (si no existe) un partner para el bot, para que los mensajes se muestren con autor "AI Assistant".
        """
        Partner = self.env['res.partner'].sudo()
        partner = Partner.search([('email', '=', AI_PARTNER_EMAIL)], limit=1)
        if partner:
            return partner
        return Partner.create({
            'name': AI_PARTNER_NAME,
            'email': AI_PARTNER_EMAIL,
            'company_type': 'person',
        })

    def _post_ai_error(self, msg, as_thread=False):
        ctx = dict(self.env.context, openai_skip=True)
        self.with_context(ctx).message_post(
            body=tools.plaintext2html(f"[OpenAI] {msg}"),
            message_type='comment',
            subtype_xmlid='mail.mt_comment',
        )
