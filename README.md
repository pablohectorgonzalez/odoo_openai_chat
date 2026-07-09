# odoo_openai_chat

Módulo de Odoo que integra un asistente de chat con OpenAI dentro del ERP. Añade capacidades conversacionales de IA a la mensajería (Discuss) de Odoo, con un runner de agentes en el servidor y extensiones en el cliente web.

## Características

Integra un asistente de chat con OpenAI en la interfaz de Odoo, un runner de agentes en el backend (services/agents_runner.py) que orquesta las llamadas al modelo, ajustes configurables desde Odoo (clave de API y opciones del modelo) vía res.config.settings, y extensiones de cliente en static/src/js. Se empaqueta como addon estándar de Odoo.

## Instalación

Copia la carpeta odoo_openai_chat en el directorio de addons de tu instancia de Odoo, activa el modo desarrollador y actualiza la lista de aplicaciones. Instala el módulo desde Aplicaciones y, en Ajustes, introduce tu API key de OpenAI.

## Requisitos

Odoo y una cuenta con API key de OpenAI.

## Autor

Pablo Héctor González Navarrete — github.com/pablohectorgonzalez
