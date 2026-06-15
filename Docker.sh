docker build -t ecg-classifier .
docker run -p 7860:7860 ecg-classifier