#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_analyzer.py (v2.0.2)
===========================

Point d'entrée unifié du projet modulaire.

Sous-commandes :
  - cli : lance l’interface console (et lui passe tous les arguments tels quels)
  - web : lance l’interface Streamlit
  - api : lance l’API FastAPI (Uvicorn)

Exemples :
  # Console (CLI)
  python crypto_analyzer.py cli --coin btc --days 90 --vs EUR

  # Web (Streamlit)
  python crypto_analyzer.py web --port 8501
  # (équivalent) python -m streamlit run interfaces/web.py

  # API (FastAPI)
  python crypto_analyzer.py api --port 8000
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys

__version__ = "2.0.2"


def _ensure_project_on_path() -> None:
    """Ajoute automatiquement le dossier du script au PYTHONPATH (imports fiables)."""
    root = os.path.abspath(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def run_cli_forward(argv: list[str] | None) -> int:
    """
    Délègue à l'interface console (interfaces/console.py).
    On lui passe les arguments 'tels quels' pour qu'elle fasse son propre argparse.
    """
    _ensure_project_on_path()
    try:
        from interfaces.console import main as console_main  # type: ignore
    except ModuleNotFoundError as e:
        print("[Erreur] Impossible d'importer interfaces.console:", e)
        print("Vérifie que interfaces/console.py existe.")
        return 2
    return console_main(argv)


def run_web(port: int = 8501, headless: bool = True) -> int:
    """
    Lance Streamlit sur interfaces/web.py via un sous-processus portable.
    """
    _ensure_project_on_path()
    script_path = os.path.join(os.path.dirname(__file__), "interfaces", "web.py")
    if not os.path.exists(script_path):
        print(f"[Erreur] Fichier introuvable : {script_path}")
        return 2

    cmd = [
        sys.executable, "-m", "streamlit", "run", script_path,
        "--server.port", str(port),
    ]
    if headless:
        cmd += ["--server.headless", "true"]

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[Erreur] Streamlit n'est pas installé.")
        print("Installe-le avec :  pip install streamlit")
        return 2


def run_api(port: int = 8000, host: str = "0.0.0.0", reload: bool = True) -> int:
    """
    Lance l'API FastAPI (interfaces/api.py) via Uvicorn.
    """
    _ensure_project_on_path()
    module_path = "interfaces.api:app"

    cmd = [
        sys.executable, "-m", "uvicorn", module_path,
        "--host", host, "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[Erreur] Uvicorn/FastAPI ne sont pas installés.")
        print("Installe-les avec :  pip install fastapi uvicorn")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto_analyzer.py",
        description="Lance l'outil en mode console, web ou API (v2.0.2)."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="mode", required=True)

    # Sous-commande CLI
    sub.add_parser("cli", help="Lancer l'interface console (args forwardés)")

    # Sous-commande WEB
    p_web = sub.add_parser("web", help="Lancer l'interface web Streamlit")
    p_web.add_argument("--port", type=int, default=8501, help="Port HTTP (par défaut 8501)")
    p_web.add_argument("--no-headless", action="store_true", help="Désactive le mode headless")

    # Sous-commande API
    p_api = sub.add_parser("api", help="Lancer l'API FastAPI (Uvicorn)")
    p_api.add_argument("--port", type=int, default=8000, help="Port HTTP (par défaut 8000)")
    p_api.add_argument("--host", type=str, default="0.0.0.0", help="Hôte (par défaut 0.0.0.0)")
    p_api.add_argument("--no-reload", action="store_true", help="Désactive le reload automatique")

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.mode == "cli":
        return run_cli_forward(unknown)

    if args.mode == "web":
        return run_web(port=args.port, headless=not args.no_headless)

    if args.mode == "api":
        return run_api(port=args.port, host=args.host, reload=not args.no_reload)

    print("[Erreur] Sous-commande inconnue.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
