import pytest
from app.tools.git_tools import grep_repo, read_file, git_blame

def test_grep_repo(mocker):
    mocker.patch('app.tools.git_tools.glob.glob', return_value=['file1.py', 'file2.py'])
    mocker.patch('app.tools.git_tools.subprocess.run', return_value=mocker.Mock(stdout='match found\n'))
    
    result = grep_repo('match', '**/*.py')
    
    assert result == ['file1.py', 'file2.py']

def test_read_file(mocker):
    mocker.patch('builtins.open', mocker.mock_open(read_data='line 1\nline 2\nline 3'))
    
    result = read_file('file1.py', 1, 2)
    
    assert result == 'line 1\nline 2\n'

def test_git_blame(mocker):
    mocker.patch('subprocess.run', return_value=mocker.Mock(stdout='commit_hash line 1\ncommit_hash line 2\n'))
    
    result = git_blame('file1.py', 1)
    
    assert result == 'commit_hash'  # Assuming the first line's blame returns the commit hash