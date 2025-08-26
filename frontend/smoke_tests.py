# frontend/smoke_tests.py
"""
SagaForge T2I Lab Frontend Smoke Tests
Test all three frontend interfaces
"""
import os
import sys
import time
import subprocess
import requests
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.shared.api_client import SagaForgeAPIClient


class FrontendSmokeTests:
    def __init__(self):
        self.api_client = SagaForgeAPIClient()
        self.results = {}

    def test_api_connection(self):
        """Test API server connection"""
        print("🔍 Testing API connection...")
        try:
            health = self.api_client.health_check()
            if health.get("status") == "ok":
                print("✅ API connection successful")
                return True
            else:
                print(f"❌ API health check failed: {health.get('message', 'Unknown')}")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {str(e)}")
            return False

    def test_pyqt_app(self):
        """Test PyQt application startup"""
        print("\n🖥️ Testing PyQt Desktop App...")
        try:
            # Test import
            from PyQt5.QtWidgets import QApplication
            from frontend.pyqt_app.main_window import MainWindow

            # Create test application
            app = QApplication([])
            window = MainWindow(self.api_client)

            # Test basic functionality
            assert window.windowTitle() == "SagaForge T2I Lab"
            assert window.api_client is not None
            assert hasattr(window, "generation_widget")
            assert hasattr(window, "lora_widget")
            assert hasattr(window, "batch_widget")
            assert hasattr(window, "training_widget")
            assert hasattr(window, "gallery_widget")

            print("✅ PyQt app initialization successful")
            print("✅ All required widgets present")

            app.quit()
            return True

        except ImportError as e:
            print(f"❌ PyQt5 import failed: {str(e)}")
            print("   Please install: pip install PyQt5")
            return False
        except Exception as e:
            print(f"❌ PyQt app test failed: {str(e)}")
            return False

    def test_gradio_app(self):
        """Test Gradio application"""
        print("\n🌐 Testing Gradio Web App...")
        try:
            # Test import
            import gradio as gr
            from frontend.gradio_app.app import create_main_interface

            # Create test interface
            interface = create_main_interface()

            # Test interface properties
            assert interface is not None
            assert hasattr(interface, "launch")

            print("✅ Gradio app creation successful")
            print("✅ Interface components loaded")

            return True

        except ImportError as e:
            print(f"❌ Gradio import failed: {str(e)}")
            print("   Please install: pip install gradio")
            return False
        except Exception as e:
            print(f"❌ Gradio app test failed: {str(e)}")
            return False

    def test_react_build(self):
        """Test React application build"""
        print("\n⚛️ Testing React App Build...")
        try:
            react_dir = Path(__file__).parent / "react_app"

            if not react_dir.exists():
                print("❌ React app directory not found")
                return False

            # Check package.json
            package_json = react_dir / "package.json"
            if not package_json.exists():
                print("❌ package.json not found")
                return False

            # Check if node_modules exists
            node_modules = react_dir / "node_modules"
            if not node_modules.exists():
                print("⚠️ node_modules not found, running npm install...")
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=react_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    print(f"❌ npm install failed: {result.stderr}")
                    return False

            # Test build
            print("Building React app...")
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=react_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print("✅ React build successful")

                # Check build output
                build_dir = react_dir / "build"
                if build_dir.exists():
                    print("✅ Build directory created")
                    return True
                else:
                    print("❌ Build directory not found")
                    return False
            else:
                print(f"❌ React build failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ React build timeout")
            return False
        except FileNotFoundError:
            print("❌ npm not found, please install Node.js")
            return False
        except Exception as e:
            print(f"❌ React build test failed: {str(e)}")
            return False

    def test_api_endpoints(self):
        """Test key API endpoints"""
        print("\n🔌 Testing API Endpoints...")

        endpoints_to_test = [
            ("/api/v1/health", "GET"),
            ("/api/v1/lora/list", "GET"),
        ]

        success_count = 0

        for endpoint, method in endpoints_to_test:
            try:
                url = f"{self.api_client.base_url}{endpoint}"
                response = requests.request(method, url, timeout=10)

                if response.status_code == 200:
                    print(f"✅ {method} {endpoint} - OK")
                    success_count += 1
                else:
                    print(f"❌ {method} {endpoint} - Status {response.status_code}")

            except Exception as e:
                print(f"❌ {method} {endpoint} - Error: {str(e)}")

        if success_count == len(endpoints_to_test):
            print("✅ All API endpoints accessible")
            return True
        else:
            print(f"⚠️ {success_count}/{len(endpoints_to_test)} API endpoints working")
            return success_count > 0

    def test_file_structure(self):
        """Test frontend file structure"""
        print("\n📁 Testing File Structure...")

        required_files = [
            "shared/api_client.py",
            "shared/api_client.js",
            "shared/constants.py",
            "pyqt_app/main.py",
            "pyqt_app/main_window.py",
            "gradio_app/app.py",
            "react_app/package.json",
            "react_app/src/App.jsx",
            "react_app/src/index.js",
        ]

        missing_files = []
        frontend_dir = Path(__file__).parent

        for file_path in required_files:
            full_path = frontend_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if not missing_files:
            print("✅ All required files present")
            return True
        else:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False

    def test_dependencies(self):
        """Test Python dependencies"""
        print("\n📦 Testing Python Dependencies...")

        required_packages = [
            "requests",
            "pathlib",
            "json",
        ]

        optional_packages = [
            ("PyQt5", "PyQt Desktop App"),
            ("gradio", "Gradio Web App"),
        ]

        # Test required packages
        missing_required = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)

        if missing_required:
            print(f"❌ Missing required packages: {', '.join(missing_required)}")
            return False
        else:
            print("✅ All required packages available")

        # Test optional packages
        for package, description in optional_packages:
            try:
                __import__(package)
                print(f"✅ {description} dependencies available")
            except ImportError:
                print(f"⚠️ {description} dependencies missing ({package})")

        return True

    def run_all_tests(self):
        """Run all smoke tests"""
        print("🚀 Starting SagaForge T2I Frontend Smoke Tests\n")

        tests = [
            ("API Connection", self.test_api_connection),
            ("File Structure", self.test_file_structure),
            ("Python Dependencies", self.test_dependencies),
            ("API Endpoints", self.test_api_endpoints),
            ("PyQt App", self.test_pyqt_app),
            ("Gradio App", self.test_gradio_app),
            ("React Build", self.test_react_build),
        ]

        results = {}

        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} test crashed: {str(e)}")
                results[test_name] = False

        # Summary
        print("\n" + "=" * 60)
        print("📊 SMOKE TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20} {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All frontend components ready!")
            return True
        elif passed >= total * 0.5:
            print("⚠️ Partial success - some components may have issues")
            return True
        else:
            print("❌ Major issues detected - please check setup")
            return False


# Frontend startup scripts
def create_startup_scripts():
    """Create startup scripts for all frontend apps"""

    scripts_dir = Path(__file__).parent / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # PyQt startup script
    pyqt_script = scripts_dir / "start_pyqt.py"
    pyqt_script.write_text(
        '''#!/usr/bin/env python
"""Start PyQt Desktop App"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from frontend.pyqt_app.main import main
    exit_code = main()
    sys.exit(exit_code)
except ImportError as e:
    print(f"Error: {e}")
    print("Please install PyQt5: pip install PyQt5")
    sys.exit(1)
except Exception as e:
    print(f"Startup failed: {e}")
    sys.exit(1)
'''
    )

    # Gradio startup script
    gradio_script = scripts_dir / "start_gradio.py"
    gradio_script.write_text(
        '''#!/usr/bin/env python
"""Start Gradio Web App"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from frontend.gradio_app.app import create_main_interface

    print("🌐 Starting Gradio Web Interface...")
    app = create_main_interface()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_tips=True,
        enable_queue=True
    )
except ImportError as e:
    print(f"Error: {e}")
    print("Please install Gradio: pip install gradio")
    sys.exit(1)
except Exception as e:
    print(f"Startup failed: {e}")
    sys.exit(1)
'''
    )

    # React startup script
    react_script = scripts_dir / "start_react.sh"
    react_script.write_text(
        '''#!/bin/bash
"""Start React Development Server"""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REACT_DIR="$SCRIPT_DIR/../react_app"

echo "⚛️ Starting React Development Server..."

# Check if React app directory exists
if [ ! -d "$REACT_DIR" ]; then
    echo "❌ React app directory not found at $REACT_DIR"
    exit 1
fi

# Navigate to React directory
cd "$REACT_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ npm install failed"
        exit 1
    fi
fi

# Start development server
echo "🚀 Launching React app at http://localhost:3000"
npm start
'''
    )

    # Make scripts executable
    import stat

    for script in [pyqt_script, gradio_script, react_script]:
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

    # Master startup script
    master_script = scripts_dir / "start_all.py"
    master_script.write_text(
        '''#!/usr/bin/env python
"""Start all frontend interfaces"""
import sys
import subprocess
import time
from pathlib import Path

def start_component(name, script_path, wait_time=2):
    print(f"🚀 Starting {name}...")
    try:
        if script_path.suffix == '.py':
            subprocess.Popen([sys.executable, str(script_path)])
        elif script_path.suffix == '.sh':
            subprocess.Popen(['bash', str(script_path)])

        time.sleep(wait_time)
        print(f"✅ {name} started")
        return True
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return False

def main():
    scripts_dir = Path(__file__).parent

    components = [
        ("Gradio Web App", scripts_dir / "start_gradio.py"),
        ("React Dev Server", scripts_dir / "start_react.sh"),
        ("PyQt Desktop App", scripts_dir / "start_pyqt.py"),
    ]

    print("🎯 Starting all SagaForge T2I frontend components...")

    success_count = 0
    for name, script_path in components:
        if start_component(name, script_path):
            success_count += 1

    print(f"\\n📊 Started {success_count}/{len(components)} components")

    if success_count > 0:
        print("""
🌟 Frontend Access URLs:
• Gradio Web:    http://localhost:7860
• React App:     http://localhost:3000
• PyQt Desktop:  Native application window

Press Ctrl+C to stop all services
""")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n👋 Shutting down all frontend services...")

    return 0 if success_count == len(components) else 1

if __name__ == "__main__":
    sys.exit(main())
'''
    )

    master_script.chmod(master_script.stat().st_mode | stat.S_IEXEC)

    print("✅ Created startup scripts:")
    for script in scripts_dir.glob("start_*"):
        print(f"   {script}")


if __name__ == "__main__":
    # Run smoke tests
    tester = FrontendSmokeTests()
    success = tester.run_all_tests()

    if success:
        print("\n🛠️ Creating startup scripts...")
        create_startup_scripts()
        print("\n🎯 Frontend setup complete!")
        print("\nQuick start commands:")
        print("python frontend/scripts/start_gradio.py     # Web interface")
        print("python frontend/scripts/start_pyqt.py       # Desktop app")
        print("bash frontend/scripts/start_react.sh        # React dev server")
        print("python frontend/scripts/start_all.py        # All interfaces")

    sys.exit(0 if success else 1)
