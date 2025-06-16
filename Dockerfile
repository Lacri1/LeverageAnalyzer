# Python 3.9 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Flask 앱이 0.0.0.0에서 실행되도록 환경 변수 설정
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# 5000 포트 노출
EXPOSE 5000

# 애플리케이션 실행
CMD ["flask", "run", "--host=0.0.0.0"]
