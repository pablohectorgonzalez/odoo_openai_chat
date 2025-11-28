// File: odoo_openai_chat/static/src/js/patch_emoji_guard.js
// Patch defensivo para evitar crash cuando emojis viene undefined en la renderización de mensajes

odoo.define('odoo_openai_chat.frontend.emoji_guard', function (require) {
    'use strict';

    // Intento de parcheo seguro de la función global _generateEmojisOnHtml (si existe)
    try {
        var targetFn = null;
        // Usualmente podría estar en window o en un namespace de frontend; intentamos lo básico
        if (typeof window._generateEmojisOnHtml === 'function') {
            targetFn = window._generateEmojisOnHtml;
        }

        if (targetFn) {
            var original = targetFn;
            window._generateEmojisOnHtml = function (input) {
                var data = input || {};
                var emojis = Array.isArray(data.emojis) ? data.emojis : [];

                // Si no hay emojis, devolver cadena vacía
                if (emojis.length === 0) {
                    return '';
                }

                // Construcción básica de HTML para cada emoji
                return emojis.map(function (e) {
                    var val = String(e);
                    return '<span class="o_emoji" title="' + val + '">' + val + '</span>';
                }).join(' ');
            };
        }
    } catch (err) {
        // En caso de cualquier fallo, no interrumpir la ejecución
        console.warn('Emoji guard patch failed (odoo_openai_chat):', err);
    }
});
