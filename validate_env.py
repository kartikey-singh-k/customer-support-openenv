#!/usr/bin/env python3
"""
Simple OpenEnv validation script for customer support environment.
Tests that the environment conforms to OpenEnv specifications.
"""
import sys
import yaml
from pathlib import Path

def validate_pyproject():
    """Validate pyproject.toml file exists and has required fields."""
    toml_path = Path("pyproject.toml")
    if not toml_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # For Python < 3.11, just check file exists
            print("✅ pyproject.toml file exists (cannot parse without tomllib)")
            return True
    
    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        
        # Check for required sections
        if "project" not in config:
            print("❌ Missing [project] section in pyproject.toml")
            return False
        
        # Check for project.scripts section (required for multi-mode deployment)
        if "project" in config and "scripts" not in config["project"]:
            print("❌ Missing [project.scripts] section in pyproject.toml")
            return False
            
        # Check if scripts section has server entry point
        scripts = config["project"].get("scripts", {})
        if "server" not in scripts:
            print("❌ Missing 'server' entry point in [project.scripts]")
            print(f"❌ Found: {list(scripts.keys())}")
            return False
        
        required_fields = ["name", "version", "description"]
        for field in required_fields:
            if field not in config["project"]:
                print(f"❌ Missing required field in [project]: {field}")
                return False
        
        print("✅ pyproject.toml validation passed")
        print(f"✅ Found server entry point: {scripts['server']}")
        return True
    
    except Exception as e:
        print(f"❌ Invalid pyproject.toml: {e}")
        return False

def validate_openenv_yaml():
    """Validate openenv.yaml file exists and has required fields."""
    yaml_path = Path("openenv.yaml")
    if not yaml_path.exists():
        print("❌ openenv.yaml not found")
        return False
    
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        required_fields = ["name", "version", "description", "entrypoint"]
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ openenv.yaml validation passed")
        return True
    
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False

def validate_entrypoint():
    """Validate that the entrypoint can be imported."""
    try:
        from src.env import CustomerSupportEnv, Action, Observation, Reward, Info
        print("✅ Environment classes imported successfully")
        
        # Test basic instantiation
        from src.tasks import EASY_TASK
        env = CustomerSupportEnv(EASY_TASK['tickets'], EASY_TASK['ground_truth'])
        obs = env.reset()
        print("✅ Environment instantiation successful")
        
        return True
    
    except Exception as e:
        print(f"❌ Environment import/instantiation failed: {e}")
        return False

def validate_structure():
    """Validate project structure."""
    required_files = [
        "openenv.yaml",
        "pyproject.toml",
        "Dockerfile", 
        "requirements.txt",
        "inference.py",
        "server/app.py",
        "server/__init__.py",
        "src/__init__.py",
        "src/env.py",
        "src/tasks.py"
    ]
    
    # Multi-mode deployment files (required for hackathon)
    multimode_files = [
        "uv.lock",
    ]
    
    # Optional files (nice to have but not critical)
    optional_files = [
        "__init__.py"  # Root package init (for proper Python package)
    ]
    
    # Check required files
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    # Check multi-mode deployment files
    for file in multimode_files:
        if not Path(file).exists():
            print(f"❌ Missing multi-mode deployment file: {file}")
            return False
    
    # Check optional files (warn but don't fail)
    for file in optional_files:
        if not Path(file).exists():
            print(f"⚠️ Missing optional file: {file} (recommended for proper Python package)")
    
    print("✅ Project structure validation passed")
    return True

def main():
    """Run all validations."""
    print("🔍 Running OpenEnv validation...")
    
    validations = [
        ("Project Structure", validate_structure),
        ("pyproject.toml", validate_pyproject),
        ("openenv.yaml", validate_openenv_yaml),
        ("Environment Import", validate_entrypoint),
    ]
    
    all_passed = True
    for name, validator in validations:
        print(f"\n📋 Validating {name}:")
        if not validator():
            all_passed = False
    
    print(f"\n{'✅ All validations passed!' if all_passed else '❌ Some validations failed'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())