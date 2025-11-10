{
    'name': 'OpenAI Chat for Discuss',
    'version': '17.0.1.3.2',
    'summary': 'Conecta OpenAI al chat de Discuss mediante /ai o chat 1:1 con un bot',
    'author': 'MetaProject',
    'license': 'LGPL-3',
    'category': 'Productivity/Discuss',
    'depends': ["base", "mail"],
    'data': [
        'views/res_config_settings_views.xml',
    ],
    "external_dependencies": {
        "python": ["requests", "openai"]  # 'openai-agents-python' opcional
    },
    'assets': {},
    'application': False,
    'installable': True,
}
