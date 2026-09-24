def grep_repo(pattern: str, path_glob: str = "**/*") -> list:
    import subprocess
    import glob

    matches = []
    for filepath in glob.glob(path_glob, recursive=True):
        try:
            result = subprocess.run(['grep', '-n', pattern, filepath], capture_output=True, text=True)
            if result.stdout:
                for line in result.stdout.splitlines():
                    line_number = line.split(':')[0]
                    matches.append({'file': filepath, 'line': int(line_number), 'content': line})
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
    return matches


def read_file(path: str, line_start: int, line_end: int) -> str:
    with open(path, 'r') as file:
        lines = file.readlines()
        return ''.join(lines[line_start - 1:line_end])


def git_blame(path: str, line: int) -> dict:
    import subprocess

    try:
        result = subprocess.run(['git', 'blame', '-L', f'{line},{line}', path], capture_output=True, text=True)
        if result.stdout:
            return {'blame_info': result.stdout.strip()}
    except Exception as e:
        print(f"Error running git blame on {path}: {e}")
    return {}