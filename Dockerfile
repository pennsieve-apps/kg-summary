FROM python:3.12.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/opt/matplotlib

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Build matplotlib's font cache into the image: the app runs as any uid with a read-only
# file system outside its step folders, so it can't build one at run time.
RUN python -c "import matplotlib.pyplot" && chmod -R a+rX /opt/matplotlib

COPY kg_summary.py /app/kg_summary.py

ENTRYPOINT ["python", "/app/kg_summary.py"]
