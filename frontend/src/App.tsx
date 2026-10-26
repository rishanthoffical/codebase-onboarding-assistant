import React, {
    useEffect,
    useRef,
    useState,
} from 'react';

import ChatPane from './components/ChatPane';
import IndexStatusPanel from './components/IndexStatusPanel';
import RepoPicker from './components/RepoPicker';
import JobTracePanel from './components/JobTracePanel';
import GuardrailBanner from './components/GuardrailBanner';

import {
    getApiErrorMessage,
    getJobStatus,
    getRepositories,
    uploadProject,
    uploadProjectFiles,
    type Repository,
} from './services/api';

import './styles.css';

interface ToolCall {
    tool?: string;
    reason?: string;
    output?: string;
}

const EXCLUDED_UPLOAD_DIRECTORIES = new Set([
    '.git',
    'node_modules',
    '.venv',
    'venv',
    'env',
    '__pycache__',
    'dist',
    'build',
    'coverage',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.tox',
    'htmlcov',
    'chroma_db',
    'chroma_cache',
    'uploaded_repositories',
    '.coverage',
    '.vscode',
    '.idea',
]);

const EXCLUDED_UPLOAD_FILES = new Set([
    '.coverage',
]);

const SENSITIVE_UPLOAD_EXTENSIONS = new Set([
    '.pem',
    '.key',
    '.p12',
    '.pfx',
    '.crt',
    '.cer',
    '.der',
    '.keystore',
]);

const App: React.FC = () => {
    const [repositories, setRepositories] =
        useState<Repository[]>([]);

    const [selectedRepo, setSelectedRepo] =
        useState<string>('local');
    const repoSelectionChangedRef = useRef(false);
    const [loadingRepositories, setLoadingRepositories] =
        useState<boolean>(true);

    const [repositoryError, setRepositoryError] =
        useState<string>('');

    const [toolCalls, setToolCalls] =
        useState<ToolCall[]>([]);

   const [importingProject, setImportingProject] =
    useState(false);

    useEffect(() => {
    console.log('IMPORTING PROJECT STATE:', importingProject);
}, [importingProject]);

  const [importMessage, setImportMessage] =
    useState('');

    

    const [overviewRepoId, setOverviewRepoId] =
        useState<string | null>(null);

    const [selectedFiles, setSelectedFiles] =
        useState<File[]>([]);

    const [excludedFileCount, setExcludedFileCount] =
        useState(0);

    const [draggingFiles, setDraggingFiles] =
        useState(false);

    const zipInput = useRef<HTMLInputElement>(null);
    const filesInput = useRef<HTMLInputElement>(null);
    const folderInput = useRef<HTMLInputElement>(null);

    useEffect(() => {
        let mounted = true;

        const loadRepositories = async () => {
            try {
                setLoadingRepositories(true);
                setRepositoryError('');

                const repos =
                    await getRepositories();

                if (!mounted) {
                    return;
                }

                setRepositories(repos);

                if (
    repos.length > 0 &&
    !repoSelectionChangedRef.current
) {
    setSelectedRepo(repos[0].repo_id);
}
            } catch (error) {
                const message = getApiErrorMessage(
                    error,
                    'Unable to load repositories from the backend.'
                );
                console.error('Failed to load repositories:', message);

                if (!mounted) {
                    return;
                }

                setRepositoryError(message);

                setRepositories([]);
                setSelectedRepo('local');
            } finally {
                if (mounted) {
                    setLoadingRepositories(false);
                }
            }
        };

        loadRepositories();

        return () => {
            mounted = false;
        };
       }, []);

    useEffect(() => {
        console.log('SELECTED REPO STATE:', selectedRepo);
    }, [selectedRepo]);

    const handleRepoChange = (
    repoId: string
) => {
    console.log('REPO SELECTED:', repoId);

    repoSelectionChangedRef.current = true;
    setSelectedRepo(repoId);

    // Clear the previous tool trace when
    // switching repositories.
    setToolCalls([]);
};

    const updateSelectedFiles = (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        const files = Array.from(event.target.files || []);
        event.target.value = '';

        applySelectedFiles(files);
    };

    const getUnsupportedArchiveMessage = (
        files: File[]
    ) => {
        const unsupported = files.find((file) =>
            /\.(rar|7z|tar|gz|bz2|xz)$/i.test(file.name)
        );

        return unsupported
            ? `${unsupported.name} is not a supported archive. Extract it and choose its folder, or convert it to ZIP.`
            : '';
    };

    const applySelectedFiles = (files: File[]) => {
        const uploadableFiles = files.filter((file) => {
            const relativePath =
                file.webkitRelativePath || file.name;
            const pathParts = relativePath
                .replace(/\\/g, '/')
                .split('/');
            const fileName = pathParts[pathParts.length - 1].toLowerCase();
            const extensionIndex = fileName.lastIndexOf('.');
            const extension =
                extensionIndex >= 0
                    ? fileName.slice(extensionIndex)
                    : '';

            return !pathParts
                .slice(0, -1)
                .some((part) =>
                    EXCLUDED_UPLOAD_DIRECTORIES.has(part.toLowerCase())
                ) &&
                !fileName.startsWith('.env') &&
                !EXCLUDED_UPLOAD_FILES.has(fileName) &&
                !SENSITIVE_UPLOAD_EXTENSIONS.has(extension);
        });

        const excludedCount = files.length - uploadableFiles.length;
        setSelectedFiles(uploadableFiles);
        setExcludedFileCount(excludedCount);

        const messages = [
            getUnsupportedArchiveMessage(uploadableFiles),
        ];

        if (excludedCount > 0) {
            messages.push(
                `${excludedCount} ignored or sensitive file${excludedCount === 1 ? ' was' : 's were'} excluded from upload.`
            );
        }

        setImportMessage(messages.filter(Boolean).join(' '));
    };

    const handleProjectUpload = async () => {
        if (selectedFiles.length === 0) {
            setImportMessage('Choose a ZIP, code files, or a project folder first.');
            return;
        }

        const unsupportedArchiveMessage =
            getUnsupportedArchiveMessage(selectedFiles);

        if (unsupportedArchiveMessage) {
            setImportMessage(unsupportedArchiveMessage);
            return;
        }

        setImportingProject(true);
        setImportMessage('Uploading project...');
        setOverviewRepoId(null);
      
        try {
            const isZip =
                selectedFiles.length === 1 &&
                selectedFiles[0].name.toLowerCase().endsWith('.zip');
            const upload = isZip
               ? await uploadProject(selectedFiles[0], selectedFiles[0].name)
               : await uploadProjectFiles(selectedFiles);

console.log(
    'UPLOAD RESPONSE JSON:',
    JSON.stringify(upload, null, 2)
);
console.log('UPLOAD JOB ID:', upload.job_id);
console.log('UPLOAD REPOSITORY ID:', upload.repository?.repo_id);
const startedAt = Date.now();

            setImportMessage('Indexing project files...');

            while (Date.now() - startedAt < 10 * 60 * 1000) {
                const job = await getJobStatus(upload.job_id);
                console.log('JOB STATUS:', job);

                if (job.status === 'completed') {
                 const repos = await getRepositories();
                 setRepositories(repos);

                console.log(
    'SETTING UPLOADED REPO:',
    upload.repository.repo_id
);

repoSelectionChangedRef.current = true;
setSelectedRepo(upload.repository.repo_id);
setOverviewRepoId(upload.repository.repo_id);
                    const excludedCount =
                        excludedFileCount + (upload.excluded_files || 0);
                    const excludedNotice = excludedCount
                        ? ` ${excludedCount} ignored or sensitive file${excludedCount === 1 ? ' was' : 's were'} excluded.`
                        : '';
                    setImportMessage(
                        `Project indexed.${excludedNotice} Generating its overview...`
                    );
                    return;
                }

                if (job.status === 'failed') {
                    throw new Error(
                        job.error || job.meta?.error || 'Project indexing failed.'
                    );
                }

                await new Promise((resolve) =>
                    window.setTimeout(resolve, 1500)
                );
            }

            throw new Error('Indexing is taking too long. Check the worker status and try again.');
        } catch (error) {
            const message = getApiErrorMessage(
                error,
                'Unable to import this project.'
            );
            console.error('Project import failed:', message);
            setImportMessage(message);
        } finally {
            setImportingProject(false);
        }
    };

    const handleDrop = (
        event: React.DragEvent<HTMLDivElement>
    ) => {
        event.preventDefault();
        setDraggingFiles(false);
        const files = Array.from(event.dataTransfer.files);
        applySelectedFiles(files);
    };

    const openFolderPicker = () => {
        folderInput.current?.click();
    };

    return (
        <div className="app">

            <GuardrailBanner />

            <RepoPicker
                repos={repositories}
                selectedRepo={selectedRepo}
                onRepoChange={handleRepoChange}
                loading={loadingRepositories}
            />

            <section className="project-intake" aria-labelledby="project-intake-title">
                <div className="project-intake-heading">
                    <div>
                        <h2 id="project-intake-title">Add a codebase</h2>
                        <span>{selectedFiles.length ? `${selectedFiles.length} file${selectedFiles.length === 1 ? '' : 's'} selected for upload${excludedFileCount ? ` (${excludedFileCount} excluded)` : ''}` : 'ZIP, files, or folder'}</span>
                    </div>
                    <button
                        type="button"
                        className="inspect-project-button"
                        disabled={importingProject || selectedFiles.length === 0 || Boolean(getUnsupportedArchiveMessage(selectedFiles))}
                        onClick={handleProjectUpload}
                    >
                        {importingProject ? 'Indexing...' : 'Index and inspect'}
                    </button>
                </div>

                <div
                    className={`project-drop-target${draggingFiles ? ' is-dragging' : ''}`}
                    onDragEnter={(event) => {
                        event.preventDefault();
                        setDraggingFiles(true);
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDragLeave={(event) => {
                        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                            setDraggingFiles(false);
                        }
                    }}
                    onDrop={handleDrop}
                >
                    <strong>Drop project files here</strong>
                    <div className="project-intake-actions">
                        <button type="button" disabled={importingProject} onClick={() => zipInput.current?.click()}>
                            Choose ZIP
                        </button>
                        <button type="button" disabled={importingProject} onClick={() => filesInput.current?.click()}>
                            Choose files
                        </button>
                        <button type="button" disabled={importingProject} onClick={openFolderPicker}>
                            Choose folder
                        </button>
                    </div>
                </div>

                <input ref={zipInput} className="project-file-input" type="file" accept=".zip,application/zip" disabled={importingProject} onChange={updateSelectedFiles} />
                <input ref={filesInput} className="project-file-input" type="file" multiple disabled={importingProject} onChange={updateSelectedFiles} />
                <input ref={folderInput} className="project-file-input" type="file" {...{ webkitdirectory: '', directory: '' }} multiple disabled={importingProject} onChange={updateSelectedFiles} />

                {selectedFiles.length > 0 && (
                    <ul className="selected-project-files">
                        {selectedFiles.slice(0, 4).map((file) => (
                            <li key={`${file.webkitRelativePath || file.name}-${file.size}`}>
                                {file.webkitRelativePath || file.name}
                            </li>
                        ))}
                        {selectedFiles.length > 4 && (
                            <li>and {selectedFiles.length - 4} more</li>
                        )}
                    </ul>
                )}
            </section>

            {importMessage && (
                <p className="project-import-status" role="status">
                    {importMessage}
                </p>
            )}

            {repositoryError && (
                <div className="repository-error">
                    {repositoryError}
                </div>
            )}

            <IndexStatusPanel
                    repoId={selectedRepo}
                     repoName={
                      repositories.find(
                         (repo) => repo.repo_id === selectedRepo
                       )?.name
                    }
                 />

            <ChatPane
                repoId={selectedRepo}
                initialQuestion={
                    overviewRepoId === selectedRepo
                        ? 'Give me a beginner-friendly overview of this codebase. Explain its purpose, languages and frameworks, entry points, important files, and main execution flow. Cite the files you use and clearly say when something is uncertain.'
                        : undefined
                }
                onToolCalls={(calls) =>
                    setToolCalls(calls)
                }
                onResponseComplete={() => {
                    if (overviewRepoId === selectedRepo) {
                        setImportMessage(
                            'Project indexed. Overview generated.'
                        );
                        setOverviewRepoId(null);
                    }
                }}
                onResponseError={() => {
                    if (overviewRepoId === selectedRepo) {
                        setImportMessage(
                            'Project indexed, but overview generation failed. See the chat for details.'
                        );
                    }
                }}
            />

            <JobTracePanel
                toolCalls={toolCalls}
            />

        </div>
    );
};

export default App;
