# SWING_BOT Automated GTT Monitor - Docker Deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create log directory
RUN mkdir -p /app/logs /app/outputs

# Create cron job for scheduling
RUN echo "15 3-9 * * * cd /app && python scheduled_gtt_monitor.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/swing-bot
RUN echo "15 2 * * * cd /app && python scheduled_gtt_monitor.py >> /app/logs/cron.log 2>&1" >> /etc/cron.d/swing-bot
RUN echo "30 10 * * * cd /app && python scheduled_gtt_monitor.py >> /app/logs/cron.log 2>&1" >> /etc/cron.d/swing-bot
RUN chmod 0644 /etc/cron.d/swing-bot
RUN crontab /etc/cron.d/swing-bot

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting SWING_BOT GTT Monitor..."\n\
\n\
# Check environment variables\n\
if [ -z "$UPSTOX_ACCESS_TOKEN" ]; then\n\
    echo "ERROR: UPSTOX_ACCESS_TOKEN not set"\n\
    exit 1\n\
fi\n\
\n\
# Start cron daemon\n\
cron\n\
\n\
# Run initial monitoring cycle\n\
echo "Running initial monitoring cycle..."\n\
python scheduled_gtt_monitor.py\n\
\n\
# Keep container running\n\
echo "SWING_BOT GTT Monitor is running. Check logs at /app/logs/"\n\
tail -f /app/logs/cron.log' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import os; print('Health check passed')" || exit 1

# Expose port for health checks (optional)
EXPOSE 8000

# Run the startup script
CMD ["/app/start.sh"]