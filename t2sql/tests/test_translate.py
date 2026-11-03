from t2sql.translate import checks


def rec(language, english, content):
    return {"language": language, "english": english, "content": content}


def test_contractions_are_not_quoted_values():
    assert checks(rec("ru", "I'm looking at the club's wins, don't count ties.", "Смотрю победы клуба, ничьи не считай.")) == []


def test_changed_quoted_value_is_flagged():
    flags = checks(rec("uz", "Which schools are in 'Amador' county?", "Qaysi maktablar shu okrugda?"))
    assert any("Amador" in f for f in flags)
    assert checks(rec("uz", "Which schools are in 'Amador' county?", "'Amador' okrugidagi maktablar?")) == []
    # an Uzbek suffix on the value is fine: the value itself is still there
    assert checks(rec("uz", "Which schools are in 'Amador' county?", "Qaysi maktablar Amadorda?")) == []


def test_missing_number_and_wrong_script_are_flagged():
    assert any("numbers missing" in f for f in checks(rec("ru", "At least 10 takers", "Как минимум десять")))
    assert any("Cyrillic" in f for f in checks(rec("uz", "How many?", "Сколько?")))
    assert any("no Cyrillic" in f for f in checks(rec("ru", "How many?", "How many?")))


def test_looks_cut_off():
    from t2sql.translate import looks_cut_off
    english = "How many patients were admitted to the hospital during the year 2023?"
    assert looks_cut_off(english, "")
    assert looks_cut_off(english, "Сколько")  # far too short
    assert looks_cut_off(english, "Сколько пациентов было принято в больницу в течение")  # no ending
    assert not looks_cut_off(english, "Сколько пациентов было принято в больницу в 2023 году?")
    assert not looks_cut_off("Top 3 stores", "Топ-3 магазина")  # the English has no end mark either
