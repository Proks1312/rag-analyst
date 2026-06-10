-- ============================================================
-- Схема finance — структурированные финансовые данные РСБУ.
--
-- Параллельный RAG-у путь для Text-to-SQL: менеджер задаёт вопрос
-- на русском, LLM пишет SELECT по этой схеме, Postgres возвращает
-- точные цифры. RAG используется для качественных вопросов и
-- контекста, finance.* — для точных финансовых выгрузок.
--
-- v1: только Формы №1 (баланс) и №2 (ОФР). Формы 3/4/6 добавим позже.
--
-- Конвенция именования колонок:
--   c_<код>_<транслит_сути>
-- Префикс c_ — чтобы код был частью имени (LLM проще писать),
-- транслит сути — чтобы имя было читаемым.
-- Все цифры — в тыс. руб. (как в выгрузке Контур.Фокус).
-- ============================================================

CREATE SCHEMA IF NOT EXISTS finance;

-- ------------------------------------------------------------
-- Компании (источники отчётности)
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS finance.company (
    id          SERIAL PRIMARY KEY,
    short_name  TEXT NOT NULL UNIQUE,    -- "АО_СИБКАБЕЛЬ" (из имени xlsx)
    full_name   TEXT,                     -- "АКЦИОНЕРНОЕ ОБЩЕСТВО «СИБКАБЕЛЬ»" (из PDF)
    ogrn        VARCHAR(15) UNIQUE,
    inn         VARCHAR(12),
    is_npo      BOOLEAN DEFAULT FALSE     -- true если у компании есть форма 6 (НКО)
);

COMMENT ON TABLE  finance.company           IS 'Справочник компаний с финансовой отчётностью РСБУ.';
COMMENT ON COLUMN finance.company.short_name IS 'Краткое имя компании, как в имени файла xlsx.';
COMMENT ON COLUMN finance.company.full_name  IS 'Полное юридическое наименование, например «АКЦИОНЕРНОЕ ОБЩЕСТВО «СИБКАБЕЛЬ»».';
COMMENT ON COLUMN finance.company.ogrn       IS 'ОГРН (13 или 15 цифр).';
COMMENT ON COLUMN finance.company.inn        IS 'ИНН (10 для юрлица или 12 для ИП).';

-- ------------------------------------------------------------
-- Форма №1 — Бухгалтерский баланс
-- Точечный срез на дату; две даты на год (Начало 01-01 и Конец 12-31).
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS finance.balance_sheet (
    company_id      INT NOT NULL REFERENCES finance.company(id) ON DELETE CASCADE,
    reporting_year  INT NOT NULL,
    period_date     DATE NOT NULL,   -- 2024-01-01 (Начало) или 2024-12-31 (Конец)

    -- АКТИВЫ — ВНЕОБОРОТНЫЕ
    c_1110_nma                NUMERIC,
    c_1120_niokr              NUMERIC,
    c_1130_npa                NUMERIC,
    c_1140_mpa                NUMERIC,
    c_1150_os                 NUMERIC,
    c_1160_dvmc               NUMERIC,
    c_1170_finvloj_dolg       NUMERIC,
    c_1180_ona                NUMERIC,
    c_1190_proch_vneobor      NUMERIC,
    c_1100_vneobor_total      NUMERIC,

    -- АКТИВЫ — ОБОРОТНЫЕ
    c_1210_zapasy             NUMERIC,
    c_1220_nds                NUMERIC,
    c_1230_debit              NUMERIC,
    c_1240_finvloj_kratk      NUMERIC,
    c_1250_dengi              NUMERIC,
    c_1260_proch_oborot       NUMERIC,
    c_1200_oborot_total       NUMERIC,

    -- КАПИТАЛ И РЕЗЕРВЫ
    c_1310_ustav_kap          NUMERIC,
    c_1320_sobst_akcii        NUMERIC,
    c_1340_pereocenka         NUMERIC,
    c_1350_dobav_kap          NUMERIC,
    c_1360_rezerv_kap         NUMERIC,
    c_1370_neraspr_pribyl     NUMERIC,
    c_1300_kapital_total      NUMERIC,

    -- ДОЛГОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА
    c_1410_dolg_zaem          NUMERIC,
    c_1420_otloj_nalog_obyaz  NUMERIC,
    c_1430_ocenoch_obyaz_dolg NUMERIC,
    c_1450_proch_dolg         NUMERIC,
    c_1400_dolgosroch_total   NUMERIC,

    -- КРАТКОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА
    c_1510_kratk_zaem         NUMERIC,
    c_1520_kred_zadolj        NUMERIC,
    c_1530_dohody_bud_per     NUMERIC,
    c_1540_ocenoch_obyaz_kratk NUMERIC,
    c_1550_proch_kratk        NUMERIC,
    c_1500_kratkosroch_total  NUMERIC,

    -- БАЛАНС
    c_1600_balans             NUMERIC,

    PRIMARY KEY (company_id, reporting_year, period_date)
);

COMMENT ON TABLE finance.balance_sheet IS
  'Бухгалтерский баланс (Форма №1 РСБУ). Все суммы — в тыс. руб. Срез на дату (Начало = 01-01 года, Конец = 12-31 года).';

-- Внеоборотные активы
COMMENT ON COLUMN finance.balance_sheet.c_1110_nma                IS 'Нематериальные активы (код 1110).';
COMMENT ON COLUMN finance.balance_sheet.c_1120_niokr              IS 'Результаты исследований и разработок, НИОКР (код 1120).';
COMMENT ON COLUMN finance.balance_sheet.c_1130_npa                IS 'Нематериальные поисковые активы (код 1130).';
COMMENT ON COLUMN finance.balance_sheet.c_1140_mpa                IS 'Материальные поисковые активы (код 1140).';
COMMENT ON COLUMN finance.balance_sheet.c_1150_os                 IS 'Основные средства (код 1150).';
COMMENT ON COLUMN finance.balance_sheet.c_1160_dvmc               IS 'Доходные вложения в материальные ценности / инвестиционная недвижимость (код 1160).';
COMMENT ON COLUMN finance.balance_sheet.c_1170_finvloj_dolg       IS 'Финансовые вложения долгосрочные (код 1170).';
COMMENT ON COLUMN finance.balance_sheet.c_1180_ona                IS 'Отложенные налоговые активы, ОНА (код 1180).';
COMMENT ON COLUMN finance.balance_sheet.c_1190_proch_vneobor      IS 'Прочие внеоборотные активы (код 1190).';
COMMENT ON COLUMN finance.balance_sheet.c_1100_vneobor_total      IS 'ИТОГО ВНЕОБОРОТНЫЕ АКТИВЫ (код 1100).';

-- Оборотные активы
COMMENT ON COLUMN finance.balance_sheet.c_1210_zapasy             IS 'Запасы (код 1210).';
COMMENT ON COLUMN finance.balance_sheet.c_1220_nds                IS 'НДС по приобретённым ценностям (код 1220).';
COMMENT ON COLUMN finance.balance_sheet.c_1230_debit              IS 'Дебиторская задолженность (код 1230).';
COMMENT ON COLUMN finance.balance_sheet.c_1240_finvloj_kratk      IS 'Финансовые вложения краткосрочные, без денежных эквивалентов (код 1240).';
COMMENT ON COLUMN finance.balance_sheet.c_1250_dengi              IS 'Денежные средства и денежные эквиваленты (код 1250).';
COMMENT ON COLUMN finance.balance_sheet.c_1260_proch_oborot       IS 'Прочие оборотные активы (код 1260).';
COMMENT ON COLUMN finance.balance_sheet.c_1200_oborot_total       IS 'ИТОГО ОБОРОТНЫЕ АКТИВЫ (код 1200).';

-- Капитал и резервы
COMMENT ON COLUMN finance.balance_sheet.c_1310_ustav_kap          IS 'Уставный капитал (код 1310).';
COMMENT ON COLUMN finance.balance_sheet.c_1320_sobst_akcii        IS 'Собственные акции, выкупленные у акционеров (код 1320).';
COMMENT ON COLUMN finance.balance_sheet.c_1340_pereocenka         IS 'Переоценка внеоборотных активов / накопленная дооценка (код 1340).';
COMMENT ON COLUMN finance.balance_sheet.c_1350_dobav_kap          IS 'Добавочный капитал без переоценки (код 1350).';
COMMENT ON COLUMN finance.balance_sheet.c_1360_rezerv_kap         IS 'Резервный капитал (код 1360).';
COMMENT ON COLUMN finance.balance_sheet.c_1370_neraspr_pribyl     IS 'Нераспределённая прибыль / непокрытый убыток (код 1370).';
COMMENT ON COLUMN finance.balance_sheet.c_1300_kapital_total      IS 'ИТОГО КАПИТАЛ И РЕЗЕРВЫ (код 1300).';

-- Долгосрочные обязательства
COMMENT ON COLUMN finance.balance_sheet.c_1410_dolg_zaem          IS 'Долгосрочные заёмные средства (код 1410).';
COMMENT ON COLUMN finance.balance_sheet.c_1420_otloj_nalog_obyaz  IS 'Отложенные налоговые обязательства, ОНО (код 1420).';
COMMENT ON COLUMN finance.balance_sheet.c_1430_ocenoch_obyaz_dolg IS 'Долгосрочные оценочные обязательства (код 1430).';
COMMENT ON COLUMN finance.balance_sheet.c_1450_proch_dolg         IS 'Прочие долгосрочные обязательства (код 1450).';
COMMENT ON COLUMN finance.balance_sheet.c_1400_dolgosroch_total   IS 'ИТОГО ДОЛГОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА (код 1400).';

-- Краткосрочные обязательства
COMMENT ON COLUMN finance.balance_sheet.c_1510_kratk_zaem         IS 'Краткосрочные заёмные средства (код 1510).';
COMMENT ON COLUMN finance.balance_sheet.c_1520_kred_zadolj        IS 'Кредиторская задолженность (код 1520).';
COMMENT ON COLUMN finance.balance_sheet.c_1530_dohody_bud_per     IS 'Доходы будущих периодов (код 1530).';
COMMENT ON COLUMN finance.balance_sheet.c_1540_ocenoch_obyaz_kratk IS 'Краткосрочные оценочные обязательства (код 1540).';
COMMENT ON COLUMN finance.balance_sheet.c_1550_proch_kratk        IS 'Прочие краткосрочные обязательства (код 1550).';
COMMENT ON COLUMN finance.balance_sheet.c_1500_kratkosroch_total  IS 'ИТОГО КРАТКОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА (код 1500).';

-- Баланс
COMMENT ON COLUMN finance.balance_sheet.c_1600_balans             IS 'БАЛАНС / валюта баланса (код 1600). Должна совпадать в активе и пассиве.';

CREATE INDEX IF NOT EXISTS idx_balance_year ON finance.balance_sheet (reporting_year);
CREATE INDEX IF NOT EXISTS idx_balance_company ON finance.balance_sheet (company_id);

-- ------------------------------------------------------------
-- Форма №2 — Отчёт о финансовых результатах (ОФР)
-- За период (год); одна строка на компанию-год.
-- В Контур.Фокус-выгрузке пара Начало/Конец, берём «Конец» = годовой итог.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS finance.income_statement (
    company_id      INT NOT NULL REFERENCES finance.company(id) ON DELETE CASCADE,
    reporting_year  INT NOT NULL,

    -- Доходы и расходы по обычным видам деятельности
    c_2110_vyruchka           NUMERIC,
    c_2120_sebest             NUMERIC,
    c_2100_valovaya_pribyl    NUMERIC,
    c_2210_komm_rashody       NUMERIC,
    c_2220_uprav_rashody      NUMERIC,
    c_2200_pribyl_prodazh     NUMERIC,

    -- Прочие доходы и расходы
    c_2310_dohody_ucastiya    NUMERIC,
    c_2320_proc_polychit      NUMERIC,
    c_2330_proc_uplat         NUMERIC,
    c_2340_proch_dohody       NUMERIC,
    c_2350_proch_rashody      NUMERIC,
    c_2300_pribyl_do_nalog    NUMERIC,

    -- Налог и финрезультат
    c_2410_nalog_pribyl       NUMERIC,
    c_2411_tek_nalog          NUMERIC,
    c_2412_otloj_nalog        NUMERIC,
    c_2421_post_nalog_obyaz   NUMERIC,
    c_2430_izmen_otloj_obyaz  NUMERIC,
    c_2450_izmen_otloj_akt    NUMERIC,
    c_2460_proch_ofr          NUMERIC,
    c_2400_chistaya_pribyl    NUMERIC,

    -- Совокупный финрезультат
    c_2500_sovokup_finrez     NUMERIC,
    c_2510_rez_pereocenki     NUMERIC,
    c_2520_rez_proch_oper     NUMERIC,
    c_2530_nalog_oper         NUMERIC,

    -- Прибыль на акцию
    c_2900_baz_pribyl_akcii   NUMERIC,
    c_2910_razv_pribyl_akcii  NUMERIC,

    PRIMARY KEY (company_id, reporting_year)
);

COMMENT ON TABLE finance.income_statement IS
  'Отчёт о финансовых результатах, ОФР (Форма №2 РСБУ). Все суммы — в тыс. руб. За календарный год (значение «Конец» из выгрузки).';

COMMENT ON COLUMN finance.income_statement.c_2110_vyruchka          IS 'Выручка (код 2110). Основной показатель доходов.';
COMMENT ON COLUMN finance.income_statement.c_2120_sebest            IS 'Себестоимость продаж / расходы по обычным видам деятельности (код 2120).';
COMMENT ON COLUMN finance.income_statement.c_2100_valovaya_pribyl   IS 'Валовая прибыль (убыток) = выручка − себестоимость (код 2100).';
COMMENT ON COLUMN finance.income_statement.c_2210_komm_rashody      IS 'Коммерческие расходы (код 2210).';
COMMENT ON COLUMN finance.income_statement.c_2220_uprav_rashody     IS 'Управленческие расходы (код 2220).';
COMMENT ON COLUMN finance.income_statement.c_2200_pribyl_prodazh    IS 'Прибыль (убыток) от продаж (код 2200).';

COMMENT ON COLUMN finance.income_statement.c_2310_dohody_ucastiya   IS 'Доходы от участия в других организациях (код 2310).';
COMMENT ON COLUMN finance.income_statement.c_2320_proc_polychit     IS 'Проценты к получению (код 2320).';
COMMENT ON COLUMN finance.income_statement.c_2330_proc_uplat        IS 'Проценты к уплате (код 2330).';
COMMENT ON COLUMN finance.income_statement.c_2340_proch_dohody      IS 'Прочие доходы (код 2340).';
COMMENT ON COLUMN finance.income_statement.c_2350_proch_rashody     IS 'Прочие расходы (код 2350).';
COMMENT ON COLUMN finance.income_statement.c_2300_pribyl_do_nalog   IS 'Прибыль (убыток) до налогообложения (код 2300).';

COMMENT ON COLUMN finance.income_statement.c_2410_nalog_pribyl      IS 'Налог на прибыль, общая сумма (код 2410).';
COMMENT ON COLUMN finance.income_statement.c_2411_tek_nalog         IS 'Текущий налог на прибыль (код 2411).';
COMMENT ON COLUMN finance.income_statement.c_2412_otloj_nalog       IS 'Отложенный налог на прибыль (код 2412).';
COMMENT ON COLUMN finance.income_statement.c_2421_post_nalog_obyaz  IS 'Постоянные налоговые обязательства / активы (код 2421).';
COMMENT ON COLUMN finance.income_statement.c_2430_izmen_otloj_obyaz IS 'Изменение отложенных налоговых обязательств (код 2430).';
COMMENT ON COLUMN finance.income_statement.c_2450_izmen_otloj_akt   IS 'Изменение отложенных налоговых активов (код 2450).';
COMMENT ON COLUMN finance.income_statement.c_2460_proch_ofr         IS 'Прочее в ОФР (код 2460).';
COMMENT ON COLUMN finance.income_statement.c_2400_chistaya_pribyl   IS 'ЧИСТАЯ ПРИБЫЛЬ (УБЫТОК) ЗА ПЕРИОД (код 2400). Bottom line ОФР.';

COMMENT ON COLUMN finance.income_statement.c_2500_sovokup_finrez    IS 'Совокупный финансовый результат периода (код 2500).';
COMMENT ON COLUMN finance.income_statement.c_2510_rez_pereocenki    IS 'Результат от переоценки внеоборотных активов (код 2510).';
COMMENT ON COLUMN finance.income_statement.c_2520_rez_proch_oper    IS 'Результат от прочих операций, не включаемых в чистую прибыль (код 2520).';
COMMENT ON COLUMN finance.income_statement.c_2530_nalog_oper        IS 'Налог на прибыль от операций, отнесённых на капитал (код 2530).';

COMMENT ON COLUMN finance.income_statement.c_2900_baz_pribyl_akcii  IS 'Базовая прибыль (убыток) на акцию, руб. (код 2900).';
COMMENT ON COLUMN finance.income_statement.c_2910_razv_pribyl_akcii IS 'Разводнённая прибыль (убыток) на акцию, руб. (код 2910).';

CREATE INDEX IF NOT EXISTS idx_income_year ON finance.income_statement (reporting_year);
CREATE INDEX IF NOT EXISTS idx_income_company ON finance.income_statement (company_id);

-- ============================================================
-- Справочник финансовых показателей (коэффициентов)
-- ============================================================
--
-- Эта таблица — единый источник правды для формул и норм по
-- коэффициентам, которые считает дашборд и SQL-маршрут.
--
-- Добавление нового показателя — INSERT в эту таблицу, и LLM в
-- sql_question.py автоматически научится его считать (через
-- fetch_indicator_glossary в промпте).
--
-- direction:
--   higher_better — больше = лучше (рост маржи, выручки)
--   lower_better  — меньше = лучше (доля долгов, доля дебиторки)
--   optimal       — должно быть в коридоре нормы (ни выше ни ниже)
-- ============================================================

CREATE TABLE IF NOT EXISTS finance.indicator_glossary (
    code            TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    description     TEXT NOT NULL,
    formula         TEXT NOT NULL,
    norm_range      TEXT,
    direction       TEXT NOT NULL CHECK (direction IN ('higher_better', 'lower_better', 'optimal')),
    notes           TEXT
);

COMMENT ON TABLE  finance.indicator_glossary IS
  'Справочник финансовых коэффициентов. SQL-маршрут (sql_question.py) подтягивает '
  'этот справочник в промпт LLM и использует formula как эталон расчёта показателя.';
COMMENT ON COLUMN finance.indicator_glossary.code         IS 'Машинное имя показателя, snake_case (например, operating_margin).';
COMMENT ON COLUMN finance.indicator_glossary.display_name IS 'Имя на дашборде / в UI (Operating Margin).';
COMMENT ON COLUMN finance.indicator_glossary.description  IS 'Расшифровка по-русски: какие статьи берутся.';
COMMENT ON COLUMN finance.indicator_glossary.formula      IS 'Эталонная формула через коды колонок finance.* (без округлений).';
COMMENT ON COLUMN finance.indicator_glossary.norm_range   IS 'Целевой коридор показателя (например, 10-15%).';
COMMENT ON COLUMN finance.indicator_glossary.direction    IS 'higher_better / lower_better / optimal.';
COMMENT ON COLUMN finance.indicator_glossary.notes        IS 'Свободные заметки: на каких компаниях не работает, нюансы расчёта.';

-- Идемпотентный seed: 5 показателей дашборда + 2 базовые маржи.
-- Можно вызывать многократно — INSERT ... ON CONFLICT обновит изменённые поля.

INSERT INTO finance.indicator_glossary (code, display_name, description, formula, norm_range, direction, notes) VALUES
  ('operating_margin',
   'Operating Margin',
   'Прибыль от продаж / Выручка. Это операционная маржа (рентабельность основной деятельности).',
   'c_2200_pribyl_prodazh / c_2110_vyruchka * 100',
   '10-15%',
   'higher_better',
   'Берётся c_2200 (прибыль от продаж), а НЕ c_2300 (прибыль до налогообложения).'),

  ('asset_turnover',
   'Asset Turnover',
   'Выручка / Активы. Сколько выручки генерирует каждый рубль активов.',
   'income.c_2110_vyruchka / balance.c_1600_balans',
   '~1.0',
   'higher_better',
   'Требует JOIN income_statement и balance_sheet по company_id и reporting_year, balance.period_date = "YYYY-12-31".'),

  ('debtors_share',
   'Debtors Share',
   'Дебиторская задолженность / Оборотные активы. Доля дебиторки в оборотных активах.',
   'c_1230_debit / c_1200_oborot_total * 100',
   '30-40%',
   'lower_better',
   'Берётся balance на конец года (period_date = "YYYY-12-31").'),

  ('quick_ratio',
   'Quick Ratio',
   'Коэффициент быстрой ликвидности: (Оборотные активы − Запасы) / Краткосрочные обязательства.',
   '(c_1200_oborot_total - c_1210_zapasy) / NULLIF(c_1500_kratkosroch_total, 0)',
   '1-2',
   'optimal',
   'Берётся balance на конец года. Значение <1 — риск ликвидности, >2 — избыточные оборотные активы.'),

  ('sga_ratio',
   'SGA Ratio',
   '(Коммерческие + Управленческие расходы) / Выручка. Доля SG&A-расходов.',
   '(c_2210_komm_rashody + c_2220_uprav_rashody) / c_2110_vyruchka * 100',
   '10-15%',
   'lower_better',
   'Высокий SGA Ratio = много расходов на содержание аппарата относительно выручки.'),

  ('net_margin',
   'Net Margin',
   'Чистая прибыль / Выручка. Рентабельность по чистой прибыли.',
   'c_2400_chistaya_pribyl / c_2110_vyruchka * 100',
   NULL,
   'higher_better',
   'Учитывает все доходы/расходы и налог на прибыль.'),

  ('pretax_margin',
   'Pretax Margin',
   'Прибыль до налогообложения / Выручка.',
   'c_2300_pribyl_do_nalog / c_2110_vyruchka * 100',
   NULL,
   'higher_better',
   'НЕ операционная маржа. Включает прочие доходы/расходы (проценты, переоценки, штрафы).')
ON CONFLICT (code) DO UPDATE SET
  display_name = EXCLUDED.display_name,
  description  = EXCLUDED.description,
  formula      = EXCLUDED.formula,
  norm_range   = EXCLUDED.norm_range,
  direction    = EXCLUDED.direction,
  notes        = EXCLUDED.notes;
