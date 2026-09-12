# COMFY TEST

Робочий простір для генеративних експериментів у ComfyUI.

## Проєкти

- [hands/](hands/) — «Руки»: 18 референсів жестів у стилі чорно-білої гравюри, щільні описи, 180 параметризованих джобів для LTX-2.5 image-to-video і сабмітер у ComfyUI. Деталі й план у [hands/PLAN.md](hands/PLAN.md).

## Запуск

Усі скрипти на Python 3, запускати через `py -3`:

```bash
py -3 hands/tools/build_jobs.py
```

```bash
py -3 hands/tools/submit_jobs.py --list
```

Адреса сервера ComfyUI береться зі змінної `COMFY_SERVER` або з дефолту в `hands/tools/submit_jobs.py`.
