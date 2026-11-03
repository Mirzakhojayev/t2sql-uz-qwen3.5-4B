import pytest

from t2sql.uz_cyrillic import russian_keyboard, to_uz_cyrillic


@pytest.mark.parametrize("latin, cyrillic", [
    ("Qaysi kitoblarning nomlari", "Қайси китобларнинг номлари"),
    ("ma'lumotlar", "маълумотлар"),
    ("o'rtacha", "ўртача"),
    ("o‘rtacha", "ўртача"),
    ("oʻrtacha", "ўртача"),
    ("O'rtacha", "Ўртача"),
    ("g'alaba", "ғалаба"),
    ("yo'q", "йўқ"),
    ("Yo'q", "Йўқ"),
    ("bo'yicha", "бўйича"),
    ("eng yuqori", "энг юқори"),
    ("Eng", "Энг"),
    ("kelgan", "келган"),
    ("yetti", "етти"),
    ("yangi", "янги"),
    ("dunyo", "дунё"),
    ("shahar", "шаҳар"),
    ("Chiqim", "Чиқим"),
    ("xodimlar", "ходимлар"),
    ("aeroport", "аэропорт"),
    ("tranzaksiya", "транзакция"),
    ("funksiyalari", "функциялари"),
    ("stansiya", "станция"),
    ("pozitsiya", "позиция"),
    ("versiya", "версия"),
    ("missiya", "миссия"),
    ("ketsa", "кетса"),
    ("sentabr oyida", "сентябр ойида"),
    ("oktabrda", "октябрда"),
    ("is'hoq", "исҳоқ"),
])
def test_words(latin, cyrillic):
    assert to_uz_cyrillic(latin) == cyrillic


def test_numbers_and_punctuation_stay():
    assert to_uz_cyrillic("10 betdan ko'proq, 2023-yil?") == "10 бетдан кўпроқ, 2023-йил?"


def test_quoted_values_stay_latin():
    assert to_uz_cyrillic("'Rock' janridagi qo'shiqlar") == "'Rock' жанридаги қўшиқлар"
    assert to_uz_cyrillic('"New York" shahri') == '"New York" шаҳри'
    assert to_uz_cyrillic("ko'p 'active' mijoz") == "кўп 'active' мижоз"


def test_names_that_the_english_also_has_stay_latin():
    english = "How many orders did Alice place in Boston?"
    assert to_uz_cyrillic("Alice Boston shahrida nechta buyurtma bergan?", english) == \
        "Alice Boston шаҳрида нечта буюртма берган?"


def test_lowercase_english_words_do_not_stop_conversion():
    assert to_uz_cyrillic("men", "men and women") == "мен"


def test_codes_stay_latin():
    assert to_uz_cyrillic("SQL va NBA o'yinlari") == "SQL ва NBA ўйинлари"
    assert to_uz_cyrillic("X25 kodi", "the code X25") == "X25 коди"


def test_english_words_the_translator_kept_stay_latin():
    assert to_uz_cyrillic("welterweight toifasi") == "welterweight тоифаси"
    assert to_uz_cyrillic("test case'ning nomi") == "тест case'нинг номи"
    assert to_uz_cyrillic("chiqim") == "чиқим"  # ch is an ordinary Uzbek letter


def test_no_latin_letters_left_in_ordinary_text():
    out = to_uz_cyrillic("O'tgan yil davomida eng ko'p sotilgan mahsulotlarning o'rtacha narxi qancha bo'lgan?")
    assert not any("a" <= c.lower() <= "z" for c in out)


def test_russian_keyboard():
    assert russian_keyboard("Ўртача қийматни ғалаба ҳақида") == "Уртача кийматни галаба хакида"
