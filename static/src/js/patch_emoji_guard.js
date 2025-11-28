odoo.define('odoo_openai_chat.emoji_guard', ['require'], function (require) {
  "use strict";

  if (typeof window._generateEmojisOnHtml === 'function') {
    const originalFn = window._generateEmojisOnHtml;
    window._generateEmojisOnHtml = function(input) {
      try {
        // Asegurar que input sea un objeto antes de intentar extraer emojis
        var data = (input && typeof input === 'object') ? input : {};
        // Evita destructuring directo; usa un fallback seguro
        var emojis = Array.isArray(data.emojis) ? data.emojis : [];
        if (!emojis || emojis.length === 0) {
          return '';
        }
        // Si hay emojis, delega al comportamiento original
        return originalFn(input);
      } catch (e) {
        // Si falla, no romper la UI
        return '';
      }
    };
  }
});
