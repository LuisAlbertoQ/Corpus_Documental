from texto_lexico import normalizar, tokenizar


def test_normalizar_quita_tildes_y_conserva_ene():
    assert normalizar("Admisión AÑO Pedagógica") == "admision año pedagogica"


def test_tokenizar_conserva_numeros_y_quita_stopwords():
    assert tokenizar("¿Cuál es el costo de pre-ingreso? S/. 120.00 en 2025") == \
        ["costo", "pre", "ingreso", "120.00", "2025"]


def test_tokenizar_articulo_numero():
    assert "53" in tokenizar("Artículo 53° nota mínima")
