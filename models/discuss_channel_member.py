# odoo_addons/odoo_openai_chat/models/discuss_channel_member.py
import logging
from odoo import models

_logger = logging.getLogger(__name__)

class DiscussChannelMember(models.Model):
    _inherit = 'discuss.channel.member'

    def write(self, vals):
        """
        Cuando el bot de OpenAI publica, evitamos tocar is_pinned y last_interest_dt
        para no competir con notify_typing y otras actualizaciones concurrentes.
        """
        if self.env.context.get('openai_skip_pin'):
            vals = dict(vals)  # copia
            removed = []
            if 'is_pinned' in vals:
                vals.pop('is_pinned', None)
                removed.append('is_pinned')
            if 'last_interest_dt' in vals:
                vals.pop('last_interest_dt', None)
                removed.append('last_interest_dt')
            if removed:
                _logger.debug("openai_skip_pin: filtrando %s en write(%s)", removed, self.ids)
        return super().write(vals)
