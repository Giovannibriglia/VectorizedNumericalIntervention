from pathlib import Path

import pytest


def run_all_tests():
    """Runs all the tests in both 'test_distributions' and 'test_estimators' directories."""
    # Get the path to the current file's directory
    current_path = Path(__file__).parent

    # Specify the directories containing your tests
    test_directories = [
        str(current_path / "test_distributions"),
        str(current_path / "test_estimators"),
    ]

    # Run pytest on the specified directories
    result = pytest.main(["-v"] + test_directories)

    # Check the return code to handle success or failure
    if result == 0:
        print("All tests passed successfully!")
    else:
        print(f"Some tests failed. Return code: {result}")


if __name__ == "__main__":
    run_all_tests()
