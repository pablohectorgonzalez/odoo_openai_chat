import logging
import re
import requests
import threading
import time
import psycopg2

import odoo
from odoo import api, fields, models, tools, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

AI_PARTNER_NAME = 'AI Assistant'
AI_PARTNER_EMAIL = 'assistant@openai.local'

def _safe_commit(cr, logger, retries=3, delay=0.2):
    """
    Intentos de commit seguros ante SerializationFailure o transacciones abortadas.
    cr: cursor/transaction de Odoo.
    logger: logger para registrar intentos.
    retries: número de reintentos permitidos.
    delay: segundos de espera entre intentos.
    """
    for attempt in range(1, retries + 1):
        try:
            cr.commit()
            return
        except psycopg2.errors.SerializationFailure as e:
            logger.warning(
                "SerializationFailure during commit (attempt %d/%d): %s",
                attempt, retries, e
            )
            time.sleep(delay)
            continue
        except psycopg2.errors.InFailedSqlTransaction as e:
            # Transacción abortada: hacer rollback y abandonar intento
            logger.error("InFailedSqlTransaction during commit (abort): %s", e)
            try:
                cr.rollback()
            except Exception:
                pass
            raise
        except Exception as e:
            logger.exception("Unexpected error during commit: %s", e)
            raise
    logger.error("Max retries exceeded for DB commit due to serialization failures.")
    raise RuntimeError("DB commit failed due to concurrent updates. Retry exhausted.")

class DiscussChannel(models.Model):
    _inherit = 'discuss.channel'  # Odoo 17

    # Contexto por canal
    openai_thread_id = fields.Char(string='OpenAI Thread ID (Assistants v2)')
    openai_agent_conversation_id = fields.Char(string='OpenAI Agent Conversation ID (Runner)')

    def message_post(self, **kwargs):
        # Evita recursión cuando el bot publica
        if self.env.context.get('openai_skip'):
            return super(DiscussChannel, self).message_post(**kwargs)

        body_html = kwargs.get('body') or ''
        body_plain = tools.html2plaintext(body_html or '').strip()

        # Post original del usuario
        message = super(DiscussChannel, self).message_post(**kwargs)

        try:
            if not self._openai_is_enabled():
                return message

            # Detectar trigger (/ai o /iaMeta)
            trigger, user_prompt = self._detect_ai_trigger(body_plain)

            # O conversación 1:1 con el bot
            if not trigger and self._is_dm_with_ai_bot():
                author_partner_id = kwargs.get('author_id') or self.env.user.partner_id.id
                ai_partner = self._get_or_create_ai_partner()
                if author_partner_id != ai_partner.id:
                    trigger = True
                    user_prompt = body_plain

            if not trigger or not user_prompt:
                return message

            cfg = self._get_openai_config()
            if not cfg.get('api_key'):
                self._post_ai_error(_('Falta el API Key de OpenAI en Ajustes'), as_thread=True)
                return message

            ai_partner = self._get_or_create_ai_partner()

            # No crear un placeholder inicial. Invocar al worker en segundo plano sin placeholder
            _logger.info("OpenAI: iniciando procesamiento asincrónico para canal=%s, prompt_len=%d",
                         self.id, len(user_prompt or ''))
            self._ai_reply_async(self.id, user_prompt, None, ai_partner.id)

        except Exception as e:
            _logger.exception('Error al procesar respuesta de OpenAI: %s', e)
            self._post_ai_error(_('Error al procesar la respuesta de OpenAI: %s') % e, as_thread=True)

        return message

    # ---------------------------
    # Worker async (thread)
    # ---------------------------

    def _ai_reply_async(self, channel_id, prompt, placeholder_message_id, ai_partner_id):
        import odoo
        from odoo import api, SUPERUSER_ID
        dbname = self.env.cr.dbname

        def _worker_once(channel_id, prompt, placeholder_message_id, ai_partner_id):
            registry = odoo.registry(dbname)
            with registry.cursor() as cr:
                env = api.Environment(cr, SUPERUSER_ID, {})
                channel = env['discuss.channel'].browse(channel_id)

                # Generar la respuesta
                try:
                    reply = channel._generate_ai_reply(prompt, exclude_message_id=placeholder_message_id)
                    if not reply:
                        reply = _("No se pudo obtener respuesta del modelo.")
                except Exception as e:
                    _logger.exception('AI worker error during generation: %s', e)
                    reply = _("No se pudo obtener respuesta del modelo.")

                # Eliminar el placeholder si existiera
                try:
                    if placeholder_message_id:
                        ph = env['mail.message'].sudo().browse(placeholder_message_id)
                        if ph.exists():
                            ph.unlink()
                except Exception as ex:
                    _logger.warning("No se pudo eliminar el placeholder %s: %s", placeholder_message_id, ex)

                # Publica el resultado
                _logger.info("AI reply length: %d", len(reply) if reply else 0)
                _logger.debug("AI reply preview: %s", (reply[:200] + '...') if reply else '""')
                try:
                    ai_msg = channel.with_context(openai_skip=True).message_post(
                        body=tools.plaintext2html(reply),
                        author_id=ai_partner_id,
                        message_type='comment',
                        subtype_xmlid='mail.mt_comment',
                    )
                    _logger.info("AI worker: published message_id=%s in channel=%s",
                                 getattr(ai_msg, 'id', None), channel_id)
                except Exception as post_ex:
                    _logger.exception("AI worker: fallo al publicar la respuesta: %s", post_ex)

                # Commit con retry ante concurrencia
                try:
                    _safe_commit(cr, _logger)
                except psycopg2.errors.InFailedSqlTransaction as e:
                    _logger.error("Commit aborted due to InFailedSqlTransaction: %s", e)
                    try:
                        cr.rollback()
                    except Exception:
                        pass
                    return
                except psycopg2.errors.SerializationFailure as e:
                    _logger.warning("SerializationFailure during commit: %s", e)
                    try:
                        cr.rollback()
                    except Exception:
                        pass
                    return
                except Exception as commit_ex:
                    _logger.exception("Commit final failed: %s", commit_ex)
                    try:
                        cr.rollback()
                    except Exception:
                        pass
                    return

        def _worker():
            _logger.info("AI worker: start canal=%s, prompt_len=%d",
                         channel_id, len(prompt) if prompt else 0)

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                registry = None
                cr = None
                try:
                    registry = odoo.registry(dbname)
                    with registry.cursor() as cr_ctx:
                        cr = cr_ctx
                        env = api.Environment(cr, SUPERUSER_ID, {})
                        # Re-fetch channel in this new environment
                        channel = env['discuss.channel'].browse(channel_id)

                        # Generar la respuesta
                        try:
                            reply = channel._generate_ai_reply(prompt, exclude_message_id=placeholder_message_id)
                            if not reply:
                                reply = _("No se pudo obtener respuesta del modelo.")
                        except Exception as e:
                            _logger.exception('AI worker error during generation: %s', e)
                            reply = _("No se pudo obtener respuesta del modelo.")

                        # Eliminar placeholder si existiera
                        try:
                            if placeholder_message_id:
                                ph = env['mail.message'].sudo().browse(placeholder_message_id)
                                if ph.exists():
                                    ph.unlink()
                        except Exception as ex:
                            _logger.warning("No se pudo eliminar el placeholder %s: %s", placeholder_message_id, ex)

                        # Publica el resultado
                        _logger.info("AI reply length: %d", len(reply) if reply else 0)
                        _logger.debug("AI reply preview: %s", (reply[:200] + '...') if reply else '""')
                        try:
                            ai_msg = channel.with_context(openai_skip=True).message_post(
                                body=tools.plaintext2html(reply),
                                author_id=ai_partner_id,
                                message_type='comment',
                                subtype_xmlid='mail.mt_comment',
                            )
                            _logger.info("AI worker: published message_id=%s in channel=%s",
                                         getattr(ai_msg, 'id', None), channel_id)
                        except Exception as post_ex:
                            _logger.exception("AI worker: fallo al publicar la respuesta: %s", post_ex)

                        # Commit con retry
                        try:
                            _safe_commit(cr, _logger)
                        except psycopg2.errors.InFailedSqlTransaction as e:
                            _logger.error("Commit aborted due to InFailedSqlTransaction: %s", e)
                            try:
                                cr.rollback()
                            except Exception:
                                pass
                            return
                        except psycopg2.errors.SerializationFailure as e:
                            _logger.warning("SerializationFailure during commit: %s", e)
                            try:
                                cr.rollback()
                            except Exception:
                                pass
                            return
                        except Exception as commit_ex:
                            _logger.exception("Commit final failed: %s", commit_ex)
                            try:
                                cr.rollback()
                            except Exception:
                                pass
                            return

                        _logger.info("AI worker: canal=%s attempt=%d finished successfully", channel_id, attempt)
                        break  # éxito, salir del loop

                except (psycopg2.errors.SerializationFailure, psycopg2.errors.InFailedSqlTransaction) as e:
                    _logger.warning("Serialization issue canal=%s attempt %d: %s", channel_id, attempt, e)
                    try:
                        if cr:
                            cr.rollback()
                    except Exception:
                        pass
                    if attempt < max_attempts:
                        time.sleep(0.2)
                        continue
                    else:
                        _logger.error("Max retries reached for canal=%s", channel_id)
                        break
                except Exception as ex:
                    _logger.exception("AI worker fatal error canal=%s: %s", channel_id, ex)
                    break

            _logger.info("AI worker: end canal=%s", channel_id)

        threading.Thread(target=_worker, name=f'openai_ai_reply_{channel_id}', daemon=True).start()

    # ---------------------------
    # Generación de respuestas y helpers
    # ---------------------------

    def _generate_ai_reply(self, prompt, exclude_message_id=None):
        ICP = self.env['ir.config_parameter'].sudo()
        mode = (ICP.get_param('openai_chat.agent_mode') or 'chat').lower()

        if mode in ('agents', 'agents_sdk'):
            try:
                from .services.agents_runner import run_agent_for_channel
                return run_agent_for_channel(self.env, self, prompt)
            except Exception as e:
                _logger.warning('Fallo Agents SDK, fallback a Assistants/Chat: %s', e)

        if mode == 'assistants':
            # Usa Assistants v2 si está seleccionado
            return self._assistants_reply(prompt) or _('No se pudo obtener respuesta del modelo.')

        # Fallback: Chat Completions
        cfg = self._get_openai_config()
        messages_payload = self._prepare_openai_chat_messages(
            user_prompt=prompt,
            ai_partner=self._get_or_create_ai_partner(),
            cfg=cfg,
            exclude_message_id=exclude_message_id
        )
        return self._call_openai_chat(messages_payload=messages_payload, cfg=cfg) or _('No se pudo obtener respuesta del modelo.')

    # ---------------------------
    # Assistants v2
    # ---------------------------
    def _assistants_reply(self, user_content):
        ICP = self.env['ir.config_parameter'].sudo()
        api_key = ICP.get_param('openai_chat.api_key')
        base_url = (ICP.get_param('openai_chat.base_url') or 'https://api.openai.com/v1').rstrip('/')
        assistant_id = ICP.get_param('openai_chat.assistant_id')
        timeout = int(ICP.get_param('openai_chat.timeout') or 60)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
        }

        thread_id = self.openai_thread_id
        if not thread_id:
            r = requests.post(f"{base_url}/threads", headers=headers, json={}, timeout=timeout)
            r.raise_for_status()
            thread_id = r.json()['id']
            self.sudo().write({'openai_thread_id': thread_id})
            self.env.cr.commit()

        r = requests.post(f"{base_url}/threads/{thread_id}/messages", headers=headers, json={"role": "user", "content": user_content}, timeout=timeout)
        r.raise_for_status()
        r = requests.post(f"{base_url}/threads/{thread_id}/runs", headers=headers, json={"assistant_id": assistant_id}, timeout=timeout)
        r.raise_for_status()
        run = r.json()
        run_id = run['id']

        import time
        start = time.time()
        while run['status'] in ('queued', 'in_progress', 'requires_action'):
            if time.time() - start > timeout:
                raise UserError(_('Timeout esperando respuesta del Assistant'))
            time.sleep(0.8)
            r = requests.get(f"{base_url}/threads/{thread_id}/runs/{run_id}", headers=headers, timeout=timeout)
            r.raise_for_status()
            run = r.json()

        if run['status'] != 'completed':
            raise UserError(_('Run terminó en estado %s') % run['status'])

        r = requests.get(f"{base_url}/threads/{thread_id}/messages", headers=headers, params={"order": "desc", "limit": 10}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        for msg in data.get('data', []):
            if msg.get('role') == 'assistant':
                parts = []
                for cpart in msg.get('content', []):
                    if cpart.get('type') == 'text':
                        parts.append(cpart['text']['value'])
                if parts:
                    return '\n'.join(parts)
        return ''

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
        if not body_plain:
            return False, ''
        m = re.match(r'^\s*/(ai|iaMeta)\s*(.*)$', body_plain, re.S | re.I)
        if m:
            return True, (m.group(2) or '').strip()
        return False, ''

    def _get_channel_partners(self):
        """Compatibilidad: usa channel_member_ids en v17; si no, channel_partner_ids."""
        if 'channel_member_ids' in self._fields:
            return self.mapped('channel_member_ids.partner_id')
        if 'channel_partner_ids' in self._fields:
            return self.channel_partner_ids
        return self.env['res.partner']

    def _is_dm_with_ai_bot(self):
        """True si el canal es un chat 1:1 entre el usuario y el bot."""
        self.ensure_one()
        if getattr(self, 'channel_type', None) != 'chat':
            return False
        partners = self._get_channel_partners()
        if len(partners) != 2:
            return False
        ai_partner = self._get_or_create_ai_partner()
        return ai_partner in partners

    # ---------------------------
    # Chat Completions (fallback)
    # ---------------------------
    def _prepare_openai_chat_messages(self, user_prompt, ai_partner, cfg, exclude_message_id=None):
        self.ensure_one()
        messages = []
        system_prompt = (cfg.get('system_prompt') or '').strip()
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        domain = [
            ('model', '=', self._name),
            ('res_id', '=', self.id),
            ('message_type', 'in', ['comment', 'email']),
            ('body', '!=', False),
        ]
        if exclude_message_id:
            domain.append(('id', '!=', exclude_message_id))

        last_msgs = self.env['mail.message'].sudo().search(domain, order='id desc', limit=cfg['context_count'])
        for msg in reversed(last_msgs):
            role = 'assistant' if (msg.author_id and msg.author_id.email == AI_PARTNER_EMAIL) else 'user'
            content = tools.html2plaintext(msg.body or '').strip()
            if not content:
                continue
            messages.append({'role': role, 'content': content})

        if user_prompt:
            messages.append({'role': 'user', 'content': user_prompt})

        return messages

    def _call_openai_chat(self, messages_payload, cfg):
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
