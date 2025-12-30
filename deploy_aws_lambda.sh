#!/bin/bash

# SWING_BOT AWS Lambda Deployment Script

set -e

echo "🚀 Deploying SWING_BOT to AWS Lambda..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Set variables
LAMBDA_FUNCTION_NAME="swing-bot-gtt-monitor"
S3_BUCKET="swing-bot-deployments-$(aws sts get-caller-identity --query Account --output text)"
REGION="ap-south-1"  # Mumbai region for better latency

# Create S3 bucket if it doesn't exist
echo "📦 Creating S3 bucket for deployments..."
aws s3 mb s3://$S3_BUCKET --region $REGION || echo "Bucket already exists"

# Create deployment package
echo "📦 Creating deployment package..."
mkdir -p deployment
cd deployment

# Copy source code
cp -r ../src ./
cp ../scheduled_gtt_monitor.py ./
cp ../requirements.txt ./
cp ../config.yaml ./

# Create lambda function
cat > lambda_function.py << 'EOF'
import json
import os
import sys
from datetime import datetime, time
import logging

# Add src to path
sys.path.append('/opt/python/src')

from scheduled_gtt_monitor import main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lambda_handler(event, context):
    """AWS Lambda handler for scheduled GTT monitoring."""
    try:
        logger.info("Lambda function triggered")
        logger.info(f"Event: {json.dumps(event)}")

        # Set environment variables from Lambda environment
        # These should be set in Lambda configuration
        required_env_vars = [
            'UPSTOX_ACCESS_TOKEN',
            'UPSTOX_API_KEY',
            'UPSTOX_API_SECRET',
            'TEAMS_WEBHOOK_URL',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]

        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': f'Missing environment variables: {missing_vars}'})
            }

        # Run the monitoring cycle
        main()

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'GTT monitoring completed successfully'})
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
EOF

# Create requirements.txt for Lambda layer
cat > lambda_requirements.txt << 'EOF'
pandas==2.1.4
numpy==1.24.3
requests==2.31.0
python-telegram-bot==20.7
boto3==1.34.34
pytz==2023.3
openpyxl==3.1.2
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4
scikit-learn==1.3.2
EOF

# Create CloudFormation template
cat > template.yaml << EOF
AWSTemplateFormatVersion: '2010-09-09'
Description: 'SWING_BOT GTT Monitor Lambda Function'

Parameters:
  FunctionName:
    Type: String
    Default: $LAMBDA_FUNCTION_NAME
    Description: Name of the Lambda function

  Environment:
    Type: String
    Default: prod
    AllowedValues: [dev, staging, prod]
    Description: Environment name

Resources:
  # IAM Role for Lambda
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '\${FunctionName}-role-\${Environment}'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: SwingBotPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: arn:aws:logs:*:*:*
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:ListBucket
                Resource:
                  - !Sub 'arn:aws:s3:::$S3_BUCKET'
                  - !Sub 'arn:aws:s3:::$S3_BUCKET/*'

  # Lambda Function
  SwingBotFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Ref FunctionName
      Runtime: python3.11
      Handler: lambda_function.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Timeout: 900  # 15 minutes
      MemorySize: 2048  # 2GB RAM
      Environment:
        Variables:
          ENVIRONMENT: !Ref Environment
          S3_BUCKET: $S3_BUCKET
      Code:
        S3Bucket: $S3_BUCKET
        S3Key: swing-bot-deployment.zip

  # EventBridge Rule for hourly schedule (9:15 AM to 3:30 PM IST)
  HourlyScheduleRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '\${FunctionName}-hourly-schedule'
      Description: 'Run SWING_BOT GTT monitor every hour from 9:15 to 15:30 IST'
      ScheduleExpression: 'cron(15 3-9 * * ? *)'  # 3:15-9:15 UTC = 9:15-15:15 IST
      State: ENABLED

  # EventBridge Rule for 8:15 AM IST
  MorningScheduleRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '\${FunctionName}-morning-schedule'
      Description: 'Run SWING_BOT GTT monitor at 8:15 AM IST'
      ScheduleExpression: 'cron(15 2 * * ? *)'  # 2:15 UTC = 8:15 IST
      State: ENABLED

  # EventBridge Rule for 4:30 PM IST
  EveningScheduleRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '\${FunctionName}-evening-schedule'
      Description: 'Run SWING_BOT GTT monitor at 4:30 PM IST'
      ScheduleExpression: 'cron(30 10 * * ? *)'  # 10:30 UTC = 16:30 IST
      State: ENABLED

  # Permissions for EventBridge to invoke Lambda
  HourlySchedulePermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref SwingBotFunction
      Action: lambda:InvokeFunction
      Principal: events.amazonaws.com
      SourceArn: !GetAtt HourlyScheduleRule.Arn

  MorningSchedulePermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref SwingBotFunction
      Action: lambda:InvokeFunction
      Principal: events.amazonaws.com
      SourceArn: !GetAtt MorningScheduleRule.Arn

  EveningSchedulePermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref SwingBotFunction
      Action: lambda:InvokeFunction
      Principal: events.amazonaws.com
      SourceArn: !GetAtt EveningScheduleRule.Arn

  # EventBridge Targets
  HourlyScheduleTarget:
    Type: AWS::Events::Rule
    DependsOn: HourlySchedulePermission
    Properties:
      Name: !Ref HourlyScheduleRule
      Targets:
        - Id: SwingBotHourlyTarget
          Arn: !GetAtt SwingBotFunction.Arn

  MorningScheduleTarget:
    Type: AWS::Events::Rule
    DependsOn: MorningSchedulePermission
    Properties:
      Name: !Ref MorningScheduleRule
      Targets:
        - Id: SwingBotMorningTarget
          Arn: !GetAtt SwingBotFunction.Arn

  EveningScheduleTarget:
    Type: AWS::Events::Rule
    DependsOn: EveningSchedulePermission
    Properties:
      Name: !Ref EveningScheduleRule
      Targets:
        - Id: SwingBotEveningTarget
          Arn: !GetAtt SwingBotFunction.Arn

Outputs:
  FunctionArn:
    Description: ARN of the Lambda function
    Value: !GetAtt SwingBotFunction.Arn
    Export:
      Name: !Sub '\${FunctionName}-arn'

  FunctionName:
    Description: Name of the Lambda function
    Value: !Ref FunctionName
    Export:
      Name: !Sub '\${FunctionName}-name'
EOF

# Package the deployment
echo "📦 Creating deployment zip..."
zip -r swing-bot-deployment.zip . -x "*.git*" "*.DS_Store" "*.pyc" "__pycache__/*"

# Upload to S3
echo "📤 Uploading deployment package to S3..."
aws s3 cp swing-bot-deployment.zip s3://$S3_BUCKET/ --region $REGION

# Deploy CloudFormation stack
echo "🚀 Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name swing-bot-gtt-monitor \
    --parameter-overrides FunctionName=$LAMBDA_FUNCTION_NAME Environment=prod \
    --capabilities CAPABILITY_IAM \
    --region $REGION

# Set environment variables
echo "🔧 Setting environment variables..."
FUNCTION_ARN=$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $REGION --query 'Configuration.FunctionArn' --output text)

# Note: Environment variables should be set manually in Lambda console for security
echo "⚠️  IMPORTANT: Set these environment variables in Lambda console:"
echo "   - UPSTOX_ACCESS_TOKEN"
echo "   - UPSTOX_API_KEY"
echo "   - UPSTOX_API_SECRET"
echo "   - TEAMS_WEBHOOK_URL (optional)"
echo "   - TELEGRAM_BOT_TOKEN (optional)"
echo "   - TELEGRAM_CHAT_ID (optional)"

echo "✅ Deployment completed!"
echo "Lambda Function ARN: $FUNCTION_ARN"
echo ""
echo "📋 Next steps:"
echo "1. Set environment variables in Lambda console"
echo "2. Test the function manually"
echo "3. Monitor CloudWatch logs"
echo "4. Check notifications in Teams/Telegram"

cd ..
rm -rf deployment