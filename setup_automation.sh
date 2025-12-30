#!/bin/bash

# SWING_BOT Automated Deployment Helper

set -e

echo "🤖 SWING_BOT Automated GTT Monitor Setup"
echo "========================================"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual credentials!"
    echo "   Required: UPSTOX_ACCESS_TOKEN, UPSTOX_API_KEY, UPSTOX_API_SECRET"
    echo "   Optional: TEAMS_WEBHOOK_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
    read -p "Press Enter after editing .env file..."
fi

echo ""
echo "Choose deployment option:"
echo "1) Docker (Recommended for self-hosting)"
echo "2) AWS Lambda (Serverless)"
echo "3) Local testing only"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🐳 Setting up Docker deployment..."

        # Check if docker and docker-compose are installed
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker not installed. Please install Docker first."
            exit 1
        fi

        if ! command -v docker-compose &> /dev/null; then
            echo "❌ Docker Compose not installed. Please install Docker Compose first."
            exit 1
        fi

        echo "🏗️  Building and starting containers..."
        docker-compose up -d --build

        echo "✅ Docker deployment complete!"
        echo ""
        echo "📊 Monitoring:"
        echo "  View logs: docker-compose logs -f swing-bot-monitor"
        echo "  Check status: docker-compose ps"
        echo "  Stop: docker-compose down"
        echo ""
        echo "📁 Persistent data is stored in ./logs and ./outputs"
        ;;

    2)
        echo "☁️  Setting up AWS Lambda deployment..."

        # Check if AWS CLI is configured
        if ! aws sts get-caller-identity &> /dev/null; then
            echo "❌ AWS CLI not configured. Run 'aws configure' first."
            exit 1
        fi

        echo "🚀 Running AWS deployment script..."
        chmod +x deploy_aws_lambda.sh
        ./deploy_aws_lambda.sh

        echo "✅ AWS Lambda deployment initiated!"
        echo ""
        echo "⚠️  IMPORTANT: Set environment variables in AWS Lambda console:"
        echo "   - UPSTOX_ACCESS_TOKEN"
        echo "   - UPSTOX_API_KEY"
        echo "   - UPSTOX_API_SECRET"
        echo "   - TEAMS_WEBHOOK_URL (optional)"
        echo "   - TELEGRAM_BOT_TOKEN (optional)"
        echo "   - TELEGRAM_CHAT_ID (optional)"
        ;;

    3)
        echo "🧪 Setting up for local testing..."

        # Check if virtual environment exists
        if [ ! -d ".venv" ]; then
            echo "📦 Creating virtual environment..."
            python -m venv .venv
        fi

        echo "🔧 Activating virtual environment and installing dependencies..."
        source .venv/Scripts/activate  # Windows
        # source .venv/bin/activate    # Linux/Mac

        pip install -r requirements.txt

        echo "✅ Local testing setup complete!"
        echo ""
        echo "🧪 Testing commands:"
        echo "  Run monitor: python scheduled_gtt_monitor.py"
        echo "  Run full pipeline: python -m src.cli orchestrate-eod"
        echo "  Check logs: tail -f logs/cron.log"
        ;;

    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "📚 Documentation: See AUTOMATED_DEPLOYMENT_README.md for details"
echo "🆘 Support: Check logs and verify environment variables"
echo ""
echo "🎉 Setup complete! Your SWING_BOT GTT Monitor is ready."