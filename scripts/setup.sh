#!/bin/bash

# SAHAYAK+ Setup Script
echo "🚀 Setting up SAHAYAK+ Disaster Management System"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/ml_models
mkdir -p logs
mkdir -p nginx/ssl

# Set permissions
chmod -R 755 data/
chmod -R 755 logs/

# Generate Django secret key
echo "🔐 Generating Django secret key..."
DJANGO_SECRET=$(python3 -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
# Database Configuration
DB_PASSWORD=sahayak_secure_2024

# Django Configuration
DJANGO_SECRET_KEY=$DJANGO_SECRET
DJANGO_DEBUG=False

# Twilio Configuration (Add your credentials)
TWILIO_SID=your_twilio_sid_here
TWILIO_TOKEN=your_twilio_token_here
TWILIO_PHONE=your_twilio_phone_here

# Firebase Configuration (Add your server key)
FCM_SERVER_KEY=your_fcm_server_key_here

# Email Configuration (Add your SMTP details)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_email_password
EMAIL_FROM=noreply@sahayak.com
EOF

echo "📧 Environment file created. Please update .env with your actual API keys."

# Initialize database
echo "🗄️ Preparing database initialization script..."
cat > init-db.sql << EOF
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_location ON auth_user USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_sos_location ON sahayak_sosrequest USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_sos_priority ON sahayak_sosrequest(priority_level, created_at);
CREATE INDEX IF NOT EXISTS idx_shelter_location ON sahayak_shelter USING GIST(location);
EOF

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check database
if docker-compose exec postgres pg_isready -U postgres; then
    echo "✅ Database is ready"
else
    echo "❌ Database is not ready"
fi

# Check backend
if curl -f http://localhost:8000/api/health/ > /dev/null 2>&1; then
    echo "✅ Backend API is ready"
else
    echo "❌ Backend API is not ready"
fi

# Check ML service
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ ML Service is ready"
else
    echo "❌ ML Service is not ready"
fi

echo "🎉 SAHAYAK+ setup complete!"
echo ""
echo "📱 Services are running at:"
echo "  • Web Dashboard: http://localhost:3000"
echo "  • Backend API: http://localhost:8000"
echo "  • ML Service: http://localhost:8001"
echo "  • Admin Panel: http://localhost:8000/admin (admin/admin123)"
echo ""
echo "📖 Next steps:"
echo "  1. Update .env file with your API keys"
echo "  2. Configure mobile apps with your server URLs"
echo "  3. Set up SSL certificates for production"
echo "  4. Configure monitoring and logging"
echo ""
echo "🆘 For support, visit: https://github.com/shahir-mohammed/sahayak-plus"
