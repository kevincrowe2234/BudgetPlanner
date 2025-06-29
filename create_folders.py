import os

def create_directory_structure():
    """Create the full directory structure for the modular budget planner app"""
    # Define all directories to create
    directories = [
        "pages",
        "components",
        "logic",
        "utils",
        "data",
        "tests"
    ]
    
    # Create each directory
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
        
    # Create __init__.py files for proper Python package structure
    for directory in ['.', 'pages', 'components', 'logic', 'utils', 'tests']:
        with open(f"{directory}/__init__.py", 'w') as f:
            f.write("# Package initialization\n")
        print(f"Created: {directory}/__init__.py")
        
    # Create .gitkeep in data directory to ensure it's tracked by git
    with open("data/.gitkeep", 'w') as f:
        pass
    print("Created: data/.gitkeep")

create_directory_structure()