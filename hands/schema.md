# Схема джобів: 24 параметри на відео

Кожен елемент `jobs.json` описує один рендер LTX-2.5 image-to-video. Порядок ключів фіксований.

| # | Параметр | Тип | Що робить |
|---|---|---|---|
| 1 | `job_id` | str | Унікальний ID, наприклад `H09_V08_fist_bump_second_beat`. Стає ім'ям файлу. |
| 2 | `ref_id` | str | Номер референса `01`..`18`. |
| 3 | `gesture` | str | Слаг жесту з `refs.json`. |
| 4 | `ref_image` | str | Шлях у папці input ComfyUI після аплоаду, `hands/09_fist_bump.jpg`. |
| 5 | `variation` | str | Слот варіації `V01`..`V10`. |
| 6 | `variation_name` | str | Назва слота, наприклад `light_sweep`. |
| 7 | `motion_type` | str | Категорія руху: `micro_hold`, `gesture_cycle`, `exit`, `fx_material`, `camera`, `lighting`, `action`, `gesture_change`, `layout`. |
| 8 | `camera_motion` | str | `static`, `dolly_in`, `orbit`, `handheld`, `tilt`. |
| 9 | `motion_intensity` | float | 0..1, скільки руху в кадрі. Зараз лише документує промпт; пізніше можна мапити на strength. |
| 10 | `end_state` | str | `lock_to_first`: останній кадр прибитий до референса (два LTXVAddGuide, кадр 0 і -1), ролик зациклюється. `free`: гайд лише на першому кадрі. |
| 11 | `prompt` | str | Повний позитивний промпт: anchor з `refs.json` + дія + style lock. |
| 12 | `audio_prompt` | str | Опис звуку. Порожній рядок означає тихе відео без аудіо-декоду. |
| 13 | `negative_prompt` | str | Негатив. При CFG 1.0 дистильована модель його ігнорує, лишаємо для недистильованих пресетів. |
| 14 | `width` | int | Ширина, кратна 32. Береться з `gen_size` референса. |
| 15 | `height` | int | Висота, кратна 32. |
| 16 | `duration_s` | int | Тривалість у секундах. Кадрів = fps × duration + 1, мусить бути 8k+1. |
| 17 | `fps` | int | Кадрів на секунду. 24 дає валідну кількість кадрів для будь-якої цілої тривалості. |
| 18 | `seed` | int | Детермінований: `20260912000 + ref*100 + variation`. |
| 19 | `sigmas_preset` | str | Ключ розкладу сигм у `submit_jobs.py`. `distilled8` = 8 кроків із шаблону. |
| 20 | `video_cfg` | float | CFG для відео-гілки LTXVDualCFGGuider. |
| 21 | `audio_cfg` | float | CFG для аудіо-гілки. |
| 22 | `img_compression` | int | LTXVPreprocess. 12 тримає зерно референса, 20 дає моделі більше свободи для розчинення. |
| 23 | `first_frame_strength` | float | Strength гайдів LTXVAddGuide. 1.0 = жорстка фіксація. |
| 24 | `output_prefix` | str | `filename_prefix` для SaveVideo: `hands/<ref>_<gesture>/<job_id>`. |

## Десять слотів варіацій

Однакова сітка для всіх 18 жестів, під кожен жест свій текст дії.

| Слот | Назва | Що відбувається | Тривалість | Кінець |
|---|---|---|---|---|
| V01 | `breath_hold` | Мікрорух: дихання, тремор сухожиль, мерехтіння зерна. Луп. | 5 с | lock |
| V02 | `perform_release` | Жест розслабляється в нейтраль і повертається точно в позу. Луп. | 6 с | lock |
| V03 | `exit_frame` | Руки виходять із кадру туди, звідки зайшли, лишаючи чистий чорний. Аутро. | 5 с | free |
| V04 | `grain_dissolve` | Руки розсипаються на стипл-точки, дрейфують і збираються назад. Луп. | 6 с | lock |
| V05 | `dolly_in` | Камера повільно наїжджає на точку інтересу, руки тримають позу. | 6 с | free |
| V06 | `orbit` | Камера обходить руки дугою, розкриваючи об'єм. | 6 с | free |
| V07 | `light_sweep` | Джерело світла проходить по руках, тіні котяться по кісточках. Луп. | 5 с | lock |
| V08 | `second_beat` | Друга дія, специфічна для жесту: кубики падають, кулаки вибухають, ОК стає клацанням. | 8 с | free |
| V09 | `interaction` | Руки взаємодіють інакше: міняються місцями, перехрещуються, торкаються. Луп. | 6 с | lock |
| V10 | `open_center` | Руки розходяться, звільняючи центр кадру під заголовок, і тримають. | 6 с | free |

## Правила

- Промпт: один абзац у теперішньому часі. Anchor, потім дія хронологічно, потім камера, світло, style lock. Без заперечень.
- `audio_prompt` дописується в кінець тексту при сабміті, тому звук описуємо як речення: `Sound: a dry knuckle thud, then silence.`
- Джерело варіацій: `src/variations_*.json`. Генератор `tools/build_jobs.py` збирає `jobs.json`, перевіряє 24 ключі, кратність 32 і 8k+1.
- Сабмітер `tools/submit_jobs.py` відтворює твій перевірений плоский граф (LoadImage, ImageScale, LTXVPreprocess, два LTXVAddGuide, SamplerCustomAdvanced, LTXVCropGuides, VAEDecodeTiled, CreateVideo, SaveVideo) і додає LTXVAudioVAEDecode, коли `audio_prompt` не порожній.
