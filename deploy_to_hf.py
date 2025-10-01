"""
Hugging Face Deployment Helper Script
Prepares the repository for Hugging Face Spaces deployment.
"""
import os
import shutil
import json

def prepare_for_deployment():
    """
    Prepare repository for Hugging Face deployment.
    """
    print("🚀 Preparing for Hugging Face Deployment")
    print("="*50)
    
    # Required files for HF deployment
    required_files = [
        'app.py',
        'requirements.txt', 
        'models.py',
        'class_names.json',
        'README.md'
    ]
    
    print("✅ Checking required files...")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (MISSING)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please ensure all required files are present.")
        return False
    
    # Check if model file exists
    model_files = ['best_model.pth', 'best_path.pth']
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"   ✓ Model file: {model_file}")
            if model_file != 'best_model.pth':
                shutil.copy(model_file, 'best_model.pth')
                print(f"   → Copied {model_file} to best_model.pth")
            model_found = True
            break
    
    if not model_found:
        print("   ⚠️  No model file found (best_model.pth or best_path.pth)")
        print("      You'll need to train a model first using:")
        print("      python train_stage2_enhanced.py")
        return False
    
    # Create .gitignore for HF Spaces
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Training results
results*/
*.log
*.out

# Data (too large for git)
data/Train/
data/Test/
*.jpg
*.jpeg
*.png

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("   ✓ Created .gitignore")
    
    # Create space configuration for HF
    space_config = """
title: Bird Species Classifier
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 3.40.0
app_file: app.py
pinned: false
"""
    
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    # Add space header if not present
    if not readme_content.startswith('---'):
        full_readme = f"---{space_config}---\n\n{readme_content}"
        with open('README.md', 'w') as f:
            f.write(full_readme)
        print("   ✓ Added HF Space configuration to README.md")
    
    print("\n🎯 Deployment Checklist:")
    print("1. ✅ All required files present")
    print("2. ✅ Model file ready")
    print("3. ✅ Configuration files created")
    print("4. ✅ .gitignore configured")
    
    print("\n📋 Next Steps for Hugging Face Deployment:")
    print("1. Create a new Space on Hugging Face:")
    print("   https://huggingface.co/new-space")
    print("2. Choose 'Gradio' as the SDK")
    print("3. Upload these files to your space:")
    for file in required_files + ['best_model.pth', '.gitignore']:
        if os.path.exists(file):
            print(f"   - {file}")
    print("4. Your app will automatically start!")
    
    print("\n🚀 Ready for deployment!")
    return True

if __name__ == "__main__":
    prepare_for_deployment()