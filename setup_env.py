#!/usr/bin/env python3
"""
Setup script to verify environment configuration for Yarl_it Legal Analyzer
"""
import os
from dotenv import load_dotenv

def verify_environment():
    """Verify that all required environment variables are set."""
    
    print("🔍 Checking environment configuration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check required variables
    required_vars = {
        "GOOGLE_API_KEY": "Google API key for Gemini models"
    }
    
    optional_vars = {
        "LOG_LEVEL": "Logging level (default: INFO)",
        "GEMINI_FLASH_MODEL": "Gemini Flash model name (default: gemini-1.5-flash)",
        "GEMINI_PRO_MODEL": "Gemini Pro model name (default: gemini-1.5-pro)"
    }
    
    missing_required = []
    
    # Check required variables
    print("\n✅ Required Environment Variables:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask API key for security
            display_value = f"{value[:8]}..." if var.endswith("_KEY") and len(value) > 8 else value
            print(f"  ✓ {var}: {display_value}")
        else:
            print(f"  ✗ {var}: NOT SET - {description}")
            missing_required.append(var)
    
    # Check optional variables
    print("\n📝 Optional Environment Variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  - {var}: Using default - {description}")
    
    # Summary
    print("\n" + "=" * 50)
    if missing_required:
        print("❌ Configuration Issues Found:")
        for var in missing_required:
            print(f"  - Please set {var} in your .env file")
        print("\n💡 Steps to fix:")
        print("  1. Copy .env.example to .env")
        print("  2. Edit .env and add your Google API key")
        print("  3. Get your API key from: https://console.cloud.google.com/")
        return False
    else:
        print("✅ Environment configuration is complete!")
        print("🚀 Your application is ready to run!")
        return True

if __name__ == "__main__":
    verify_environment()
