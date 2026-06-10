-- ============================================================
-- Миграция: добавить отраслевой сегмент в finance.company
-- ============================================================
-- Источник: «Справочник клиентов» из 1С продаж Brusit+ за 2023-2024.
-- Сопоставление прошло fuzzy-матчингом по нормализованному имени.
-- 23/23 совпадений. Колонка segment текстовая (10 уникальных значений),
-- не вынесена в отдельный справочник — пока сегментов мало и они
-- стабильные, отдельная таблица не нужна.
--
-- ВАЖНО: short_name в БД хранится С ПРОБЕЛАМИ (см. SELECT short_name
-- в Supabase: «АО СИБКАБЕЛЬ», «ООО Билдэкс», ...). Локальный Docker
-- может содержать тот же формат — но на всякий случай вместо exact
-- WHERE используем ILIKE-патерн (работает и с подчёркиваниями, и с
-- пробелами, и независимо от регистра).
-- ============================================================

-- 1) Колонка + индекс
ALTER TABLE finance.company
    ADD COLUMN IF NOT EXISTS segment TEXT;

CREATE INDEX IF NOT EXISTS idx_company_segment
    ON finance.company(segment);

COMMENT ON COLUMN finance.company.segment IS
    'Отраслевой сегмент (по справочнику клиентов 1С Brusit+). '
    'Возможные значения: Кабельное производство, Компаунды/Полимеры, '
    'Кормовые добавки, Молочная промышленность (раскислитель), АКП, '
    'Удобрения/Агрохимикаты, Металлургия/Огнеупорные материалы/Флюс, '
    'РТИ, Дистрибьютер, Переработчики (B2B).';


-- 2) На случай частично-залитой предыдущей миграции — сбрасываем.
UPDATE finance.company SET segment = NULL;


-- 3) Заполняем segment через ILIKE — терпит пробел/подчёркивание/регистр.
UPDATE finance.company SET segment = CASE
    WHEN short_name ILIKE '%русхимсеть%'                          THEN 'Дистрибьютер'
    WHEN short_name ILIKE '%сибкабель%'                           THEN 'Кабельное производство'
    WHEN short_name ILIKE '%экз%'                                 THEN 'Кабельное производство'
    WHEN short_name ILIKE '%гагаринконсерв%'                      THEN 'Молочная промышленность (раскислитель)'
    WHEN short_name ILIKE '%курскрезино%'                         THEN 'РТИ'
    WHEN short_name ILIKE '%хёс%' OR short_name ILIKE '%хес%'     THEN 'Кормовые добавки'
    WHEN short_name ILIKE '%алпина%'                              THEN 'Переработчики (B2B)'
    WHEN short_name ILIKE '%билдэкс%' OR short_name ILIKE '%билдекс%' THEN 'АКП'
    WHEN short_name ILIKE '%кужель%'                              THEN 'Удобрения/Агрохимикаты'
    WHEN short_name ILIKE '%молочная фабрика%' OR short_name ILIKE '%молочная_фабрика%' THEN 'Молочная промышленность (раскислитель)'
    WHEN short_name ILIKE '%мустанг%ступино%'                     THEN 'Кормовые добавки'
    WHEN short_name ILIKE '%проффасад%'                           THEN 'АКП'
    WHEN short_name ILIKE '%птицефабрика%гурьевск%'               THEN 'Кормовые добавки'
    WHEN short_name ILIKE '%сибалюкс%'                            THEN 'АКП'
    WHEN short_name ILIKE '%полиметалл%'                          THEN 'Металлургия/Огнеупорные материалы/Флюс'
    WHEN short_name ILIKE '%техкорм%'                             THEN 'Кормовые добавки'
    WHEN short_name ILIKE '%дионис%'                              THEN 'Молочная промышленность (раскислитель)'
    WHEN short_name ILIKE '%ул%полимер%композит%'                 THEN 'Компаунды/Полимеры'
    WHEN short_name ILIKE '%уральский%завод%пластификатор%'       THEN 'Компаунды/Полимеры'
    WHEN short_name ILIKE '%фурмановское%'                        THEN 'Удобрения/Агрохимикаты'
    WHEN short_name ILIKE '%эко%компаунд%групп%'                  THEN 'Компаунды/Полимеры'
    WHEN short_name ILIKE '%нлмк%'                                THEN 'Металлургия/Огнеупорные материалы/Флюс'
    WHEN short_name ILIKE '%северсталь%'                          THEN 'Металлургия/Огнеупорные материалы/Флюс'
    ELSE segment
END;


-- 4) Проверка — должно вернуть 0 непокрытых
SELECT short_name
FROM finance.company
WHERE segment IS NULL
ORDER BY short_name;

-- 5) Сводка по сегментам
SELECT segment, COUNT(*) AS companies
FROM finance.company
GROUP BY segment
ORDER BY companies DESC, segment;
