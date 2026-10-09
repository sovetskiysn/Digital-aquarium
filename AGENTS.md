# AGENTS.md

Этот файл — инструкция для ИИ-агентов (Claude Code, Codex и любых других),
работающих в этом репозитории. Он описывает, что здесь лежит, как устроено
окружение и каких соглашений придерживаться при правках. Если что-то в
репозитории изменилось — обновляйте этот файл вместе с изменением.

## О репозитории

Рабочая папка проекта **Digital-aquarium**: эволюция агентов в виртуальной
среде — децентрализованная мультиагентная дискретная среда, агенты в виде
простой нейросети, обучение классическим генетическим алгоритмом.

Папка разделена на две половины: код и тексты.

## Структура

```
.
├── AGENTS.md            # этот файл
├── README.md            # описание проекта и запуск через uv
├── .gitignore           # единый для обеих половин, пути с префиксами
├── .vscode/             # настройки редактора и рекомендованные расширения
├── python_project/      # код симуляции
│   ├── main.py          # точка входа: окно pygame с симуляцией
│   ├── applications.py
│   ├── engine/          # среда, сущности, «мозги» агентов
│   ├── visualization/   # отрисовка pygame, параметры, текстуры, тема UI
│   ├── experiments/     # .xlsx с результатами прогонов
│   ├── assets/          # Thesis.pdf, article.docx, preza.pptx, графики, демо
│   ├── .python-version  # Python 3.12 для uv
│   ├── pyproject.toml   # зависимости и индекс PyTorch для uv
│   ├── uv.lock          # точные версии зависимостей
│   └── requirements.txt # прежний список зависимостей
└── latex_documents/     # LaTeX-документы по проекту (пока пусто)
    └── Makefile         # make clean — убрать артефакты ручной сборки
```

Git-репозиторий находится в корне рабочей папки, `origin` —
`https://github.com/sovetskiysn/Digital-aquarium.git`.
Все изменения делаем в ветке `new_article`; `main` не меняем.

## python_project

- Python 3.12, зависимости для uv в `pyproject.toml` (pygame, pygame_gui,
  numpy, torch, pandas); точные версии закреплены в `uv.lock`.
- Установка: `cd python_project && uv sync --locked`; запуск: `uv run main.py`.
- На текущей машине нет GPU: `torch==2.6.0+cpu` для uv привязан к индексу
  `https://download.pytorch.org/whl/cpu` через `tool.uv.sources`.
  Вариант CUDA 12.6 оставлен в комментариях в `pyproject.toml` на будущее.
- `requirements.txt` сохранён в UTF-16 — `pip install -r` на нём падает, пока
  файл не перекодирован в UTF-8.
- Виртуальное окружение — `python_project/.venv/` (в `.gitignore`).
- `main.py` открывает окно pygame. На сервере по SSH для этого нужен
  X11-проброс (`echo $DISPLAY` не должен быть пустым); без окна — запуск через
  `xvfb-run uv run main.py`.

## latex_documents: LaTeX Workshop + Docker

Сборка идёт через расширение **LaTeX Workshop** (`James-Yu.latex-workshop`,
рекомендовано в `.vscode/extensions.json`). Локальный TeX Live не нужен —
компиляция запускается в Docker-образе `texlive/texlive:TL2025-historic` при
каждом сохранении `.tex`. Тег `historic` не двигается, поэтому сборка
воспроизводима; **не меняйте его на `latest`** без явной причины.

Один документ = одна подпапка `latex_documents/<имя>/` со своим `.tex`.

Артефакты сборки (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.out`) уходят в
`.build/` рядом с документом (`latex-workshop.latex.auxDir`). PDF и
`.synctex.gz` остаются на месте — иначе не работают просмотрщик и SyncTeX.
Сборка руками мимо LaTeX Workshop кладёт артефакты рядом с `.tex`; они скрыты
в дереве VS Code и исключены из гита, удаляются через `make clean` в
`latex_documents/`.

## Соглашения

- **Единые конфиги в корне** — `.gitignore` и подобные файлы держим в одном
  экземпляре в корне, секции половин разделены комментариями, пути
  префиксуются (`python_project/**`, `latex_documents/**`).
- **Не удалять и не «прибирать» незнакомые файлы** по своей инициативе —
  только по явной просьбе.
- **Git-коммиты** — не добавлять `Co-Authored-By: Claude ...` в сообщения
  коммитов.
