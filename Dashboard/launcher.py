"""
ShopBis Dashboard Launcher
==========================
Simple launcher script to start the Streamlit dashboard
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the ShopBis Analytics Dashboard"""

    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent
    app_file = dashboard_dir / "app.py"

    # Check if app.py exists
    if not app_file.exists():
        print("❌ Error: app.py not found!")
        print(f"   Expected location: {app_file}")
        sys.exit(1)

    # Change to dashboard directory
    os.chdir(dashboard_dir)

    print("🛍️  ShopBis Analytics Dashboard")
    print("=" * 50)
    print("📊 Starting Streamlit server...")
    print("=" * 50)
    print()
    print("✅ Dashboard will open in your browser automatically")
    print("🌐 URL: http://localhost:8501")
    print()
    print("⚠️  To stop the server: Press Ctrl+C")
    print("=" * 50)
    print()

    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=false"
        ], check=True)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("👋 Dashboard stopped. Thank you for using ShopBis!")
        print("=" * 50)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\n💡 Try running: streamlit run app.py")
        sys.exit(1)

    except FileNotFoundError:
        print("\n❌ Error: Streamlit not installed!")
        print("\n💡 Install dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
