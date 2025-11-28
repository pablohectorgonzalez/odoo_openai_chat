// File: odoo_openai_chat/static/src/js/patch_emoji_guard_safe.js
// Parche defensivo: envolver la generación de emojis para evitar crashes cuando input es undefined.

odoo.define('odoo_openai_chat.frontend.emoji_guard_safe', function (require) {
    'use strict';

    try {
        if (typeof window._generateEmojisOnHtml === 'function') {
            var originalFn = window._generateEmojisOnHtml;
            window._generateEmojisOnHtml = function (input) {
                try {
                    var data = input || {};
                    // Si no hay emojis o la propiedad no es arreglo, devolvemos cadena vacía
                    var emojis = Array.isArray(data.emojis) ? data.emojis : [];
                    if (emojis.length === 0) {
                        return '';
                    }
                    // Si hay emojis, delegamos en la función original
                    return originalFn(input);
                } catch (e) {
                    // Proteger contra cualquier fallo en renderizar emojis
                    return '';
                }
            };
        }
    } catch (e) {
        console.warn('OpenAI Chat: emoji guard patch failed (safe version)', e);
    }
});
