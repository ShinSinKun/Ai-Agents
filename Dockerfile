FROM python:3.9-slim

# Установка необходимых зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочего каталога
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .

# Создание виртуального окружения
RUN python -m venv venv

# Активация виртуального окружения и установка зависимостей
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остального приложения
COPY . .

# Обеспечение прав на выполнение скрипта
RUN chmod +x start.sh

# Запуск приложения
CMD ["./start.sh"]
