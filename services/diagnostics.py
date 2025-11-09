# services/diagnostics.py
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import socket
import asyncio
import logging
import platform
import inspect
from typing import Dict, Any

import requests
from importlib import metadata as importlib_metadata

from odoo import tools, _

_logger = logging.getLogger(__name__)

def _mask_secret(s: str) -> str:
    if not s:
        return ''
    if len(s) <= 8:
        return '*' * len(s)
    return f"{s[:4]}***{s[-4:]}"

def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not-installed"

def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def _resolve_host(url: str):
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if not p.hostname:
            return None, "no-host"
        ip = socket.gethostbyname(p.hostname)
        return ip, None
    except Exception as e:
        return None, str(e)

def _supports_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("o4") or m.startswith("gpt-4.1")

def _extract_output_text(resp) -> str:
    # Para openai.responses
    txt = getattr(resp, "output_text", None)
    if txt:
        return str(txt)
    # Para Agents Runner results
    if hasattr(resp, "final_output_as"):
        try:
            return resp.final_output_as(str)
        except Exception:
            pass
    for attr in ("final_output", "output"):
        v = getattr(resp, attr, None)
        if v:
            return str(v)
    return ""

def run_deep_diagnostics(env, prompt='Di "pong".', channel_id=None, timeout=25) -> Dict[str, Any]:
    """
    Retorna un dict con diagnóstico profundo.
    Ejecuta:
    - Validación de config y entorno
    - Ping HTTP a base_url
    - Llamada OpenAI (models.list, retrieve del modelo, responses.create)
    - Verificación de Vector Stores
    - Test de filestore + SQLiteSession
    - Test del Agents Runner en 4 pasos (con/sin tools y con fallback de modelo)
    """
    ICP = env['ir.config_parameter'].sudo()
    api_key = ICP.get_param('openai_chat.api_key') or ''
    base_url = (ICP.get_param('openai_chat.base_url') or 'https://api.openai.com/v1').rstrip('/')
    agent_mode = (ICP.get_param('openai_chat.agent_mode') or 'chat').lower()
    agent_model = ICP.get_param('openai_chat.agent_model') or ''
    chat_model = ICP.get_param('openai_chat.model') or ''
    assistant_id = ICP.get_param('openai_chat.assistant_id') or ''
    vec_ids_str = ICP.get_param('openai_chat.agent_vector_store_ids') or ''
    vec_ids = [v.strip() for v in vec_ids_str.split(',') if v.strip()]
    fallback_model = 'o4-mini'

    # Canal para sesión
    channel = None
    if channel_id:
        channel = env['discuss.channel'].browse(int(channel_id))
    if not channel:
        channel = env['discuss.channel'].search([], limit=1)

    # Info base
    report: Dict[str, Any] = {
        "summary": {},
        "env_info": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "user": getattr(os, "geteuid", lambda: None)(),
            "venv": os.environ.get("VIRTUAL_ENV") or "",
            "packages": {
                "openai": _pkg_version("openai"),
                "openai-agents": _pkg_version("openai-agents"),
                "httpx": _pkg_version("httpx"),
                "requests": _pkg_version("requests"),
                "pydantic": _pkg_version("pydantic"),
            },
            "env_vars": {
                "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL") or "",
                "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE") or "",
                "HTTP_PROXY": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "",
                "HTTPS_PROXY": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "",
            },
        },
        "config": {
            "base_url": base_url,
            "api_key_present": bool(api_key),
            "api_key_sample": _mask_secret(api_key),
            "agent_mode": agent_mode,
            "agent_model": agent_model,
            "chat_model": chat_model,
            "assistant_id_present": bool(assistant_id),
            "vector_store_ids_count": len(vec_ids),
        },
        "checks": [],
    }

    def add_check(name, status, duration=None, details=None, error=None, extra=None):
        report["checks"].append({
            "name": name,
            "status": status,  # ok|warn|error
            "duration": duration,
            "details": _jsonable(details),
            "error": (str(error) if error else None),
            "extra": _jsonable(extra) if extra is not None else None,
        })

    # 1) Resolver host
    ip, err = _resolve_host(base_url)
    add_check("dns_resolve_base_url", "ok" if ip else "error", details={"ip": ip}, error=err)

    # 2) Ping HTTP base_url (sin auth)
    try:
        t0 = time.time()
        r = requests.get(base_url, timeout=10)
        add_check("http_ping_base_url", "ok" if r.status_code in (200, 401, 404) else "warn",
                  duration=round(time.time() - t0, 3),
                  details={"status_code": r.status_code})
    except Exception as e:
        add_check("http_ping_base_url", "error", error=e)

    # 3) OpenAI client básico
    try:
        t0 = time.time()
        from openai import OpenAI, __version__ as openai_version
        client = OpenAI(api_key=api_key, base_url=base_url)
        models_list = client.models.list()
        add_check("openai_models_list", "ok", duration=round(time.time() - t0, 3),
                  details={"count": len(models_list.data), "openai_version": openai_version})
    except Exception as e:
        add_check("openai_models_list", "error", error=e)

    # 4) Modelo configurado (agent_model o chat_model)
    target_model = agent_model or chat_model or ""
    if target_model:
        try:
            t0 = time.time()
            m = client.models.retrieve(target_model)
            add_check("openai_model_retrieve", "ok", duration=round(time.time() - t0, 3),
                      details={"id": getattr(m, "id", target_model)})
        except Exception as e:
            add_check("openai_model_retrieve", "error", error=e, details={"id": target_model})
    else:
        add_check("openai_model_retrieve", "warn", details="no hay modelo configurado; se usará fallback")

    # 5) Llamada minimal de Responses (sin Agents) para validar API
    test_model = target_model or fallback_model
    try:
        t0 = time.time()
        resp = client.responses.create(model=test_model, input=prompt)
        text = _extract_output_text(resp)
        add_check("openai_responses_create", "ok" if bool(text) else "warn",
                  duration=round(time.time() - t0, 3),
                  details={"model": test_model, "output": text[:200]})
    except Exception as e:
        add_check("openai_responses_create", "error", error=e, details={"model": test_model})

    # 6) Vector stores
    if vec_ids:
        for vs in vec_ids:
            try:
                t0 = time.time()
                vs_obj = client.vector_stores.retrieve(vs)
                add_check(f"vector_store_retrieve:{vs}", "ok", duration=round(time.time() - t0, 3),
                          details={"id": getattr(vs_obj, "id", vs)})
            except AttributeError as e:
                add_check(f"vector_store_retrieve:{vs}", "warn", error="client.vector_stores no disponible en esta versión de openai")
                break
            except Exception as e:
                add_check(f"vector_store_retrieve:{vs}", "error", error=e)
    else:
        add_check("vector_stores", "warn", details="sin vector_store_ids (FileSearchTool inactivo)")

    # 7) Filestore + SQLite path
    try:
        filestore_dir = tools.config.filestore(env.cr.dbname)
        agents_dir = os.path.join(filestore_dir, 'openai_agents')
        os.makedirs(agents_dir, exist_ok=True)
        test_file = os.path.join(agents_dir, 'diag.tmp')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
        details = {"filestore": filestore_dir, "agents_dir": agents_dir}
        add_check("filestore_write", "ok", details=details)
    except Exception as e:
        add_check("filestore_write", "error", error=e)

    # 8) Agents SDK import + Runner.run signature
    try:
        from agents import FileSearchTool, Agent, ModelSettings, Runner, RunConfig, trace, SQLiteSession
        from openai.types.shared.reasoning import Reasoning
        sig = None
        try:
            sig = str(inspect.signature(Runner.run))
        except Exception:
            sig = "unknown"
        add_check("agents_import", "ok", details={"Runner.run.signature": sig})
    except Exception as e:
        add_check("agents_import", "error", error=e)

    # 9) Session SQLite
    try:
        filestore_dir = tools.config.filestore(env.cr.dbname)
        session_path = os.path.join(filestore_dir, 'openai_agents', f'channel_{channel.id if channel else 0}.sqlite3')
        # No forzamos abrir; se abrirá en Runner.run. Verificamos ruta.
        add_check("agents_session_path", "ok", details={"session_path": session_path})
    except Exception as e:
        add_check("agents_session_path", "error", error=e)

    # 10) Agents Runner en cascada (4 pasos)
    def _new_agent(model=None, with_tools=True):
        tools_list = []
        if with_tools and vec_ids:
            tools_list.append(FileSearchTool(vector_store_ids=vec_ids))
        ms_kwargs = dict(store=True)
        mdl = model or (agent_model or chat_model or fallback_model)
        if _supports_reasoning(mdl):
            try:
                ms_kwargs['reasoning'] = Reasoning(effort="high", summary="auto")
            except Exception:
                pass
        agent = Agent(
            name="Diag Agent",
            instructions="Eres un agente de diagnóstico; responde exactamente: pong",
            model=mdl,
            tools=tools_list,
            model_settings=ModelSettings(**ms_kwargs),
        )
        return agent

    async def _run_agent(agent, session):
        with trace("Diag Agents Run"):
            kw = dict(
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                session=session,
                run_config=RunConfig(trace_metadata={"__trace_source__": "diag", "channel_id": str(channel.id if channel else 0)}),
            )
            # request_timeout si la firma lo soporta
            try:
                return await Runner.run(agent, request_timeout=timeout, **kw)
            except TypeError:
                return await Runner.run(agent, **kw)

    def _run_async(coro):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.stop()
            except Exception:
                pass
            loop.close()
            asyncio.set_event_loop(None)

    os.environ['OPENAI_API_KEY'] = api_key or os.environ.get('OPENAI_API_KEY', '')
    os.environ['OPENAI_BASE_URL'] = base_url
    os.environ['OPENAI_API_BASE'] = base_url

    try:
        # Prepara session
        filestore_dir = tools.config.filestore(env.cr.dbname)
        agents_dir = os.path.join(filestore_dir, 'openai_agents')
        os.makedirs(agents_dir, exist_ok=True)
        session_path = os.path.join(agents_dir, f'channel_{channel.id if channel else 0}.sqlite3')
        session = SQLiteSession(session_path)

        steps = [
            ("agents_step_1_model_cfg_with_tools", None, True),
            ("agents_step_2_model_cfg_no_tools", None, False),
            ("agents_step_3_fallback_with_tools", fallback_model, True),
            ("agents_step_4_fallback_no_tools", fallback_model, False),
        ]
        for name, mdl, with_tools in steps:
            try:
                agent = _new_agent(model=mdl, with_tools=with_tools)
                t0 = time.time()
                result = _run_async(_run_agent(agent, session))
                text = _extract_output_text(result)
                dur = round(time.time() - t0, 3)
                status = "ok" if text else "warn"
                add_check(name, status, duration=dur, details={"model": agent.model, "with_tools": bool(vec_ids) and with_tools, "output": text[:200]})
            except Exception as e:
                add_check(name, "error", error=e, details={"model": mdl or (agent_model or chat_model or fallback_model), "with_tools": with_tools})
    except Exception as e:
        add_check("agents_runner_block", "error", error=e)

    # Resumen
    errors = [c for c in report["checks"] if c["status"] == "error"]
    warns = [c for c in report["checks"] if c["status"] == "warn"]
    report["summary"] = {
        "ok": len(errors) == 0,
        "errors": len(errors),
        "warnings": len(warns),
    }
    return report


def format_report(report: Dict[str, Any]) -> str:
    lines = []
    s = report.get("summary", {})
    lines.append(f"OK: {s.get('ok')} | errors: {s.get('errors')} | warnings: {s.get('warnings')}")
    cfg = report.get("config", {})
    lines.append(
        f"base_url={cfg.get('base_url')}, api_key_present={cfg.get('api_key_present')}, "
        f"agent_mode={cfg.get('agent_mode')}, agent_model={cfg.get('agent_model')}, "
        f"vec_store_count={cfg.get('vector_store_ids_count')}"
    )
    for c in report.get("checks", []):
        d = c.get("details") or {}
        msg = f"- {c['name']}: {c['status']}"
        if c.get("duration") is not None:
            msg += f" ({c['duration']}s)"
        if isinstance(d, dict):
            model = d.get("model")
            if model:
                msg += f" [model={model}]"
            status_code = d.get("status_code")
            if status_code is not None:
                msg += f" [status={status_code}]"
            out = d.get("output") or ""
            if out:
                out80 = out[:80]
                # Sanitizar antes de interpolar (evita backslashes en f-string)
                out80 = out80.replace("\n", " ").replace("\r", " ")
                msg += f" out={out80}"
        if c.get("error"):
            # Aquí no usamos backslashes en expresiones; es seguro
            msg += " ERR={}".format(str(c["error"])[:200])
        lines.append(msg)
    return "\n".join(lines)
