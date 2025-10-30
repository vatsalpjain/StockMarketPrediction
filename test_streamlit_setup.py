"""
Test script to validate Streamlit setup
Run this before launching the Streamlit app to check for issues
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)
    
    required_packages = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'plotly': 'Plotly',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'yfinance': 'yfinance',
        'torch': 'PyTorch'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20s} - OK")
        except ImportError as e:
            print(f"❌ {name:20s} - MISSING")
            failed.append(name)
    
    print()
    
    if failed:
        print(f"⚠️  Missing packages: {', '.join(failed)}")
        print(f"   Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages installed!")
        return True

def test_project_structure():
    """Test if all required files and directories exist"""
    print("\n" + "=" * 60)
    print("Testing Project Structure...")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    required_files = [
        'app.py',
        'main.py',
        'requirements.txt',
        'streamlit_pages/__init__.py',
        'streamlit_pages/single_step_page.py',
        'streamlit_pages/multi_horizon_page.py',
        'streamlit_pages/results_page.py',
        'streamlit_pages/comparison_page.py',
        'pipelines/single_step_pipeline.py',
        'pipelines/multi_horizon_pipeline.py',
        'config/settings.py'
    ]
    
    missing = []
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path:50s} - OK")
        else:
            print(f"❌ {file_path:50s} - MISSING")
            missing.append(file_path)
    
    print()
    
    if missing:
        print(f"⚠️  Missing files: {', '.join(missing)}")
        return False
    else:
        print("✅ All required files present!")
        return True

def test_streamlit_pages():
    """Test if streamlit pages can be imported"""
    print("\n" + "=" * 60)
    print("Testing Streamlit Pages...")
    print("=" * 60)
    
    pages = [
        'streamlit_pages.single_step_page',
        'streamlit_pages.multi_horizon_page',
        'streamlit_pages.results_page',
        'streamlit_pages.comparison_page'
    ]
    
    failed = []
    
    for page in pages:
        try:
            __import__(page)
            print(f"✅ {page:50s} - OK")
        except Exception as e:
            print(f"❌ {page:50s} - ERROR: {str(e)[:30]}")
            failed.append(page)
    
    print()
    
    if failed:
        print(f"⚠️  Failed to import: {', '.join(failed)}")
        return False
    else:
        print("✅ All pages can be imported!")
        return True

def test_pipelines():
    """Test if pipelines can be imported"""
    print("\n" + "=" * 60)
    print("Testing Pipelines...")
    print("=" * 60)
    
    try:
        from pipelines.single_step_pipeline import SingleStepPipeline
        print(f"✅ SingleStepPipeline - OK")
    except Exception as e:
        print(f"❌ SingleStepPipeline - ERROR: {str(e)}")
        return False
    
    try:
        from pipelines.multi_horizon_pipeline import MultiHorizonPipeline
        print(f"✅ MultiHorizonPipeline - OK")
    except Exception as e:
        print(f"❌ MultiHorizonPipeline - ERROR: {str(e)}")
        return False
    
    print()
    print("✅ All pipelines can be imported!")
    return True

def test_config():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("Testing Configuration...")
    print("=" * 60)
    
    try:
        from config import settings
        print(f"✅ Settings module - OK")
        print(f"   - PREDICTION_HORIZONS: {settings.PREDICTION_HORIZONS}")
        print(f"   - DEFAULT_PERIOD: {settings.DEFAULT_PERIOD}")
        print(f"   - FEATURE_COLS: {len(settings.FEATURE_COLS)} features")
    except Exception as e:
        print(f"❌ Settings module - ERROR: {str(e)}")
        return False
    
    # Check for API keys (optional)
    try:
        from config import api_keys
        if hasattr(api_keys, 'FINNHUB_API_KEY'):
            print(f"✅ Finnhub API key configured")
        else:
            print(f"ℹ️  Finnhub API key not configured (optional)")
    except:
        print(f"ℹ️  API keys file not found (optional)")
    
    print()
    print("✅ Configuration OK!")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("STREAMLIT SETUP VALIDATION")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Streamlit Pages", test_streamlit_pages()))
    results.append(("Pipelines", test_pipelines()))
    results.append(("Configuration", test_config()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now run the Streamlit app:")
        print("  streamlit run app.py")
        print("\nOr on Windows, double-click: run_streamlit.bat")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running the app.")
        print("Common fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check file paths and imports")
    
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
