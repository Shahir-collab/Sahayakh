#!/bin/bash

# SAHAYAK+ Mobile App Setup Script
echo "📱 Setting up SAHAYAK+ Mobile Applications"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if React Native CLI is installed
if ! command -v react-native &> /dev/null; then
    echo "📦 Installing React Native CLI..."
    npm install -g react-native-cli
fi

# Setup mobile directory
cd mobile/

# Install dependencies
echo "📦 Installing mobile app dependencies..."
npm install

# iOS setup (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Setting up iOS dependencies..."
    cd ios/
    pod install
    cd ..
fi

# Android setup
echo "🤖 Setting up Android configuration..."

# Create local.properties for Android
cat > android/local.properties << EOF
sdk.dir=$ANDROID_HOME
EOF

# Setup Firebase configuration files
echo "🔥 Firebase setup required:"
echo "  1. Download google-services.json from Firebase Console"
echo "  2. Place it in android/app/ directory"
echo "  3. Download GoogleService-Info.plist for iOS"
echo "  4. Place it in ios/ directory"

# Create sample config files
mkdir -p src/config/

cat > src/config/api.js << EOF
// API Configuration
const config = {
  API_BASE_URL: __DEV__ 
    ? 'http://10.0.2.2:8000/api'  // Android emulator
    : 'https://api.sahayak.com/api',
  
  WS_BASE_URL: __DEV__
    ? 'ws://10.0.2.2:8000/ws'
    : 'wss://api.sahayak.com/ws',
    
  ML_BASE_URL: __DEV__
    ? 'http://10.0.2.2:8001'
    : 'https://api.sahayak.com/ml'
};

export default config;
EOF

echo "📱 Mobile app setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Configure Firebase (google-services.json & GoogleService-Info.plist)"
echo "  2. Update API URLs in src/config/api.js"
echo "  3. Run 'react-native run-android' for Android"
echo "  4. Run 'react-native run-ios' for iOS (macOS only)"
echo ""
echo "🔧 Development commands:"
echo "  • npm start - Start Metro bundler"
echo "  • npm run android - Run on Android"
echo "  • npm run ios - Run on iOS"
echo "  • npm test - Run tests"