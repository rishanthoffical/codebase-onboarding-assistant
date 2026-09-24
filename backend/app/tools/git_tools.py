import glob
import subprocess


def grep_repo(pattern: str, repo_path: str = "."):
    """
    Search repository files for a pattern.

    Returns the files matched by the repository glob.
    """
    files = glob.glob(repo_path)

    return files


def read_file(path: str, start_line: int = 1, end_line: int = None):
    """
    Read selected lines from a file and return them as a string.
    """
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if end_line is None:
        end_line = len(lines)

    return "".join(lines[start_line - 1:end_line])


def git_blame(path: str, line: int):
    """
    Get the commit hash responsible for a specific line.
    """
    result = subprocess.run(
        [
            "git",
            "blame",
            "-L",
            f"{line},{line}",
            path
        ],
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout.strip()

    if not output:
        return ""

    # Git blame normally starts with the commit hash.
    return output.split()[0]