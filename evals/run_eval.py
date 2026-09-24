import os
import subprocess

def run_eval(repo_path, questions):
    results = {}
    
    for question in questions:
        # Simulate running a question against the codebase
        command = f"python3 -m your_evaluation_module --repo {repo_path} --question '{question}'"
        try:
            output = subprocess.check_output(command, shell=True, text=True)
            results[question] = output
        except subprocess.CalledProcessError as e:
            results[question] = f"Error: {e.output}"
    
    return results

if __name__ == "__main__":
    # Example usage
    repo_path = os.getenv("REPO_PATH", "/path/to/your/repo")
    questions = [
        "How does authentication work in this service?",
        "Where can I find the data processing logic?",
    ]
    
    evaluation_results = run_eval(repo_path, questions)
    for question, result in evaluation_results.items():
        print(f"Question: {question}\nResult: {result}\n")