#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
texto_lexico.py — Normalización y tokenización léxica para BM25.

IMPORTANTE: OE5 debe tokenizar las preguntas con EXACTAMENTE esta función
(copiar este archivo a oe5_chatbot_upeu/backend/ en la fase de OE5). Si cambia,
hay que regenerar bm25_corpus_<version>.json con 08_indice_bm25.py.
"""

import re
import unicodedata

VERSION_TOKENIZADOR = "es-v1"

STOPWORDS_ES = frozenset("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquello aquellos aqui
asi aun cada como con contra cual cuales cualquier cuando de del desde donde dos e el ella ellas ello
ellos en entre era eran es esa esas ese eso esos esta estas este esto estos fue fueron ha han hasta hay
la las le les lo los mas me mi mis mucho muy ni no nos o os otra otras otro otros para pero poco por
porque que quien quienes se sea sean segun ser si sido sin sobre son su sus tal tambien te tiene tienen
todo todos tu tus u un una unas uno unos y ya
""".split())

# Números con decimales/separadores ("120.00", "2025", "5.3.1") se conservan como un token.
_RE_TOKEN = re.compile(r"[a-z0-9ñ]+(?:[.,][0-9]+)*")


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes (conserva la ñ)."""
    texto = texto.lower().replace("ñ", "\x00")
    texto = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return texto.replace("\x00", "ñ")


def tokenizar(texto: str) -> list[str]:
    return [t for t in _RE_TOKEN.findall(normalizar(texto)) if t not in STOPWORDS_ES and (len(t) > 1 or t.isdigit())]
