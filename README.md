# Smart City Traffic Monitor

Wizyjny system analizy ruchu drogowego w czasie rzeczywistym. Aplikacja pobiera strumień wideo z kamer miejskich (m.in. z platformy YouTube), wykrywa pojazdy i analizuje ich ruch, prezentując wyniki na interaktywnym panelu.

## Funkcjonalności
* **Śledzenie trajektorii:** Wykrywanie pojazdów przy użyciu modelu YOLOv8s i śledzenie ich ścieżek na ekranie.
* **Filtracja tła:** System ignoruje obiekty statyczne (np. zaparkowane samochody), zliczając wyłącznie pojazdy w ruchu (wymagany minimalny dystans przemieszczenia).
* **Detekcja zatorów:** Monitorowanie zagęszczenia ruchu i automatyczne generowanie alertów w przypadku przekroczenia zdefiniowanego progu.
* **Optymalizacja wydajności:** Wykorzystanie wielowątkowości (osobny wątek do pobierania wideo) oraz asynchronicznego zapisu danych do plików CSV w celu eliminacji opóźnień (I/O blocking).
* **Klasyfikacja:** Podział zliczanych obiektów na kategorie (samochody osobowe, ciężarówki, autobusy, motocykle).

## Technologie
* **AI / Computer Vision:** Python, OpenCV, Ultralytics (YOLOv8)
* **Interfejs / Backend:** Streamlit, Pandas
* **Przetwarzanie strumieni:** Streamlink, Threading

## Uruchomienie lokalne
1. Sklonuj repozytorium.
2. Zainstaluj wymagane pakiety: `pip install -r requirements.txt`
3. Uruchom aplikację: `streamlit run smart_city.py`
