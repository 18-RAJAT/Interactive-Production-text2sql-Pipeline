import React, { useState, useRef, useEffect, useCallback } from "react";
import { Plus, ChevronDown, ArrowUp, X, FileText, Loader2, Check, Archive } from "lucide-react";

/* --- ICONS --- */
export const Icons = {
    Logo: (props: React.SVGProps<SVGSVGElement>) => (
        <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="presentation" {...props}>
            <defs>
                <ellipse id="petal-pair" cx="100" cy="100" rx="90" ry="22" />
            </defs>
            <g fill="#D46B4F" fillRule="evenodd">
                <use href="#petal-pair" transform="rotate(0 100 100)" />
                <use href="#petal-pair" transform="rotate(45 100 100)" />
                <use href="#petal-pair" transform="rotate(90 100 100)" />
                <use href="#petal-pair" transform="rotate(135 100 100)" />
            </g>
        </svg>
    ),
    Plus: Plus,
    Thinking: (props: React.SVGProps<SVGSVGElement>) => <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" {...props}><path d="M10.3857 2.50977C14.3486 2.71054 17.5 5.98724 17.5 10C17.5 14.1421 14.1421 17.5 10 17.5C5.85786 17.5 2.5 14.1421 2.5 10C2.5 9.72386 2.72386 9.5 3 9.5C3.27614 9.5 3.5 9.72386 3.5 10C3.5 13.5899 6.41015 16.5 10 16.5C13.5899 16.5 16.5 13.5899 16.5 10C16.5 6.5225 13.7691 3.68312 10.335 3.50879L10 3.5L9.89941 3.49023C9.67145 3.44371 9.5 3.24171 9.5 3C9.5 2.72386 9.72386 2.5 10 2.5L10.3857 2.50977ZM10 5.5C10.2761 5.5 10.5 5.72386 10.5 6V9.69043L13.2236 11.0527C13.4706 11.1762 13.5708 11.4766 13.4473 11.7236C13.3392 11.9397 13.0957 12.0435 12.8711 11.9834L12.7764 11.9473L9.77637 10.4473C9.60698 10.3626 9.5 10.1894 9.5 10V6C9.5 5.72386 9.72386 5.5 10 5.5ZM3.66211 6.94141C4.0273 6.94159 4.32303 7.23735 4.32324 7.60254C4.32324 7.96791 4.02743 8.26446 3.66211 8.26465C3.29663 8.26465 3 7.96802 3 7.60254C3.00021 7.23723 3.29676 6.94141 3.66211 6.94141ZM4.95605 4.29395C5.32146 4.29404 5.61719 4.59063 5.61719 4.95605C5.6171 5.3214 5.3214 5.61709 4.95605 5.61719C4.59063 5.61719 4.29403 5.32146 4.29395 4.95605C4.29395 4.59057 4.59057 4.29395 4.95605 4.29395ZM7.60254 3C7.96802 3 8.26465 3.29663 8.26465 3.66211C8.26446 4.02743 7.96791 4.32324 7.60254 4.32324C7.23736 4.32302 6.94159 4.0273 6.94141 3.66211C6.94141 3.29676 7.23724 3.00022 7.60254 3Z"></path></svg>,
    SelectArrow: ChevronDown,
    ArrowUp: ArrowUp,
    X: X,
    FileText: FileText,
    Loader2: Loader2,
    Check: Check,
    Archive: Archive,
    Clock: (props: React.SVGProps<SVGSVGElement>) => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor" {...props}><path d="M10.3857 2.50977C14.3486 2.71054 17.5 5.98724 17.5 10C17.5 14.1421 14.1421 17.5 10 17.5C5.85786 17.5 2.5 14.1421 2.5 10C2.5 9.72386 2.72386 9.5 3 9.5C3.27614 9.5 3.5 9.72386 3.5 10C3.5 13.5899 6.41015 16.5 10 16.5C13.5899 16.5 16.5 13.5899 16.5 10C16.5 6.5225 13.7691 3.68312 10.335 3.50879L10 3.5L9.89941 3.49023C9.67145 3.44371 9.5 3.24171 9.5 3C9.5 2.72386 9.72386 2.5 10 2.5L10.3857 2.50977ZM10 5.5C10.2761 5.5 10.5 5.72386 10.5 6V9.69043L13.2236 11.0527C13.4706 11.1762 13.5708 11.4766 13.4473 11.7236C13.3392 11.9397 13.0957 12.0435 12.8711 11.9834L12.7764 11.9473L9.77637 10.4473C9.60698 10.3626 9.5 10.1894 9.5 10V6C9.5 5.72386 9.72386 5.5 10 5.5ZM3.66211 6.94141C4.0273 6.94159 4.32303 7.23735 4.32324 7.60254C4.32324 7.96791 4.02743 8.26446 3.66211 8.26465C3.29663 8.26465 3 7.96802 3 7.60254C3.00021 7.23723 3.29676 6.94141 3.66211 6.94141ZM4.95605 4.29395C5.32146 4.29404 5.61719 4.59063 5.61719 4.95605C5.6171 5.3214 5.3214 5.61709 4.95605 5.61719C4.59063 5.61719 4.29403 5.32146 4.29395 4.95605C4.29395 4.59057 4.59057 4.29395 4.95605 4.29395ZM7.60254 3C7.96802 3 8.26465 3.29663 8.26465 3.66211C8.26446 4.02743 7.96791 4.32324 7.60254 4.32324C7.23736 4.32302 6.94159 4.0273 6.94141 3.66211C6.94141 3.29676 7.23724 3.00022 7.60254 3Z"></path></svg>,
};

/* --- UTILS --- */
const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

/* --- COMPONENTS --- */

export interface AttachedFile {
    id: string;
    file: File;
    type: string;
    preview: string | null;
    uploadStatus: string;
    content?: string;
}

interface FilePreviewCardProps {
    file: AttachedFile;
    onRemove: (id: string) => void;
}

const FilePreviewCard: React.FC<FilePreviewCardProps> = ({ file, onRemove }) => {
    const isImage = file.type.startsWith("image/") && file.preview;

    return (
        <div className="relative group flex-shrink-0 w-24 h-24 rounded-xl overflow-hidden border border-claude-bg-300 bg-claude-bg-200 animate-claude-fade-in transition-all hover:border-claude-text-400">
            {isImage ? (
                <div className="w-full h-full relative">
                    <img src={file.preview!} alt={file.file.name} className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors" />
                </div>
            ) : (
                <div className="w-full h-full p-3 flex flex-col justify-between">
                    <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-claude-bg-300 rounded">
                            <Icons.FileText className="w-4 h-4 text-claude-text-300" />
                        </div>
                        <span className="text-[10px] font-medium text-claude-text-400 uppercase tracking-wider truncate">
                            {file.file.name.split('.').pop()}
                        </span>
                    </div>
                    <div className="space-y-0.5">
                        <p className="text-xs font-medium text-claude-text-200 truncate" title={file.file.name}>
                            {file.file.name}
                        </p>
                        <p className="text-[10px] text-claude-text-500">
                            {formatFileSize(file.file.size)}
                        </p>
                    </div>
                </div>
            )}

            <button
                onClick={() => onRemove(file.id)}
                className="absolute top-1 right-1 p-1 bg-black/50 hover:bg-black/70 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity"
            >
                <Icons.X className="w-3 h-3" />
            </button>

            {file.uploadStatus === 'uploading' && (
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <Icons.Loader2 className="w-5 h-5 text-white animate-spin" />
                </div>
            )}
        </div>
    );
};

interface Model {
    id: string;
    name: string;
    description: string;
    badge?: string;
}

interface ModelSelectorProps {
    models: Model[];
    selectedModel: string;
    onSelect: (modelId: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ models, selectedModel, onSelect }) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const currentModel = models.find(m => m.id === selectedModel) || models[0];

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`inline-flex items-center justify-center relative shrink-0 transition font-base duration-300 ease-[cubic-bezier(0.165,0.85,0.45,1)] h-8 rounded-xl px-3 min-w-[4rem] active:scale-[0.98] whitespace-nowrap !text-xs pl-2.5 pr-2 gap-1 
                ${isOpen
                        ? 'bg-claude-bg-200 text-claude-text-100 dark:bg-[#454540] dark:text-[#ECECEC]'
                        : 'text-claude-text-300 hover:text-claude-text-200 hover:bg-claude-bg-200 dark:text-[#B4B4B4] dark:hover:text-[#ECECEC] dark:hover:bg-[#454540]'}`}
            >
                <div className="font-ui inline-flex gap-[3px] text-[14px] h-[14px] leading-none items-baseline">
                    <div className="flex items-center gap-[4px]">
                        <div className="whitespace-nowrap select-none font-medium">{currentModel.name}</div>
                    </div>
                </div>
                <div className="flex items-center justify-center opacity-75" style={{ width: '20px', height: '20px' }}>
                    <Icons.SelectArrow className={`shrink-0 opacity-75 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
                </div>
            </button>

            {isOpen && (
                <div className="absolute bottom-full right-0 mb-2 w-[260px] bg-white dark:bg-[#212121] border border-[#DDDDDD] dark:border-[#30302E] rounded-2xl shadow-2xl overflow-hidden z-50 flex flex-col p-1.5 animate-claude-fade-in origin-bottom-right">
                    {models.map(model => (
                        <button
                            key={model.id}
                            onClick={() => {
                                onSelect(model.id);
                                setIsOpen(false);
                            }}
                            className="w-full text-left px-3 py-2.5 rounded-xl flex items-start justify-between group transition-colors hover:bg-claude-bg-200 dark:hover:bg-[#30302E]"
                        >
                            <div className="flex flex-col gap-0.5">
                                <div className="flex items-center gap-2">
                                    <span className="text-[13px] font-semibold text-claude-text-100 dark:text-[#ECECEC]">
                                        {model.name}
                                    </span>
                                    {model.badge && (
                                        <span className={`px-1.5 py-[1px] rounded-full text-[10px] font-medium border ${model.badge === 'Upgrade'
                                            ? 'border-blue-200 text-blue-600 bg-white dark:border-blue-500/30 dark:text-blue-400 dark:bg-blue-500/10'
                                            : 'border-claude-bg-300 text-claude-text-300'
                                            }`}>
                                            {model.badge}
                                        </span>
                                    )}
                                </div>
                                <span className="text-[11px] text-claude-text-300 dark:text-[#999999]">
                                    {model.description}
                                </span>
                            </div>
                            {selectedModel === model.id && (
                                <Icons.Check className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-1" />
                            )}
                        </button>
                    ))}

                    <div className="h-px bg-claude-bg-300 dark:bg-[#30302E] my-1 mx-2" />

                    <button className="w-full text-left px-3 py-2.5 rounded-xl flex items-center justify-between group transition-colors hover:bg-claude-bg-200 dark:hover:bg-[#30302E] text-claude-text-100 dark:text-[#ECECEC]">
                        <span className="text-[13px] font-semibold">More models</span>
                        <Icons.SelectArrow className="w-4 h-4 -rotate-90 text-claude-text-300 dark:text-[#999999]" />
                    </button>
                </div>
            )}
        </div>
    );
};

export interface ClaudeChatInputProps {
    onSendMessage: (data: {
        message: string;
        files: AttachedFile[];
        model: string;
        isThinkingEnabled: boolean;
    }) => void;
}

export const ClaudeChatInput: React.FC<ClaudeChatInputProps> = ({ onSendMessage }) => {
    const [message, setMessage] = useState("");
    const [files, setFiles] = useState<AttachedFile[]>([]);
    const [isDragging, setIsDragging] = useState(false);
    const [selectedModel, setSelectedModel] = useState("sonnet-4.5");
    const [isThinkingEnabled, setIsThinkingEnabled] = useState(false);

    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const models = [
        { id: "opus-4.5", name: "Opus 4.5", description: "Most capable for complex work" },
        { id: "sonnet-4.5", name: "Sonnet 4.5", description: "Best for everyday tasks" },
        { id: "haiku-4.5", name: "Haiku 4.5", description: "Fastest for quick answers" }
    ];

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 384) + "px";
        }
    }, [message]);

    const handleFiles = useCallback((newFilesList: FileList | File[]) => {
        const newFiles = Array.from(newFilesList).map(file => {
            const isImage = file.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(file.name);
            return {
                id: Math.random().toString(36).substr(2, 9),
                file,
                type: isImage ? 'image/unknown' : (file.type || 'application/octet-stream'),
                preview: isImage ? URL.createObjectURL(file) : null,
                uploadStatus: 'pending'
            };
        });

        setFiles(prev => [...prev, ...newFiles]);

        setMessage(prev => {
            if (prev) return prev;
            if (newFiles.length === 1) {
                const f = newFiles[0];
                if (f.type.startsWith('image/')) return "Analyzed image...";
                return "Analyzed document...";
            }
            return `Analyzed ${newFiles.length} files...`;
        });

        newFiles.forEach(f => {
            setTimeout(() => {
                setFiles(prev => prev.map(p => p.id === f.id ? { ...p, uploadStatus: 'complete' } : p));
            }, 800 + Math.random() * 1000);
        });
    }, []);

    const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
    const onDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };
    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
    };

    const handlePaste = (e: React.ClipboardEvent) => {
        const items = e.clipboardData.items;
        const pastedFiles: File[] = [];
        for (let i = 0; i < items.length; i++) {
            if (items[i].kind === 'file') {
                const file = items[i].getAsFile();
                if (file) pastedFiles.push(file);
            }
        }

        if (pastedFiles.length > 0) {
            e.preventDefault();
            handleFiles(pastedFiles);
        }
    };

    const handleSend = () => {
        if (!message.trim() && files.length === 0) return;
        onSendMessage({ message, files, model: selectedModel, isThinkingEnabled });
        setMessage("");
        setFiles([]);
        if (textareaRef.current) textareaRef.current.style.height = 'auto';
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const hasContent = message.trim() || files.length > 0;

    return (
        <div
            className="relative w-full max-w-2xl mx-auto transition-all duration-300 font-sans"
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
        >
            <div className="!box-content flex flex-col mx-2 md:mx-0 items-stretch transition-all duration-200 relative z-10 rounded-2xl cursor-text border border-claude-bg-300 dark:border-transparent shadow-[0_0_15px_rgba(0,0,0,0.08)] hover:shadow-[0_0_20px_rgba(0,0,0,0.12)] focus-within:shadow-[0_0_25px_rgba(0,0,0,0.15)] bg-white dark:bg-[#30302E] font-sans antialiased">
                <div className="flex flex-col px-3 pt-3 pb-2 gap-2">

                    {files.length > 0 && (
                        <div className="flex gap-3 overflow-x-auto custom-scrollbar pb-2 px-1">
                            {files.map(file => (
                                <FilePreviewCard
                                    key={file.id}
                                    file={file}
                                    onRemove={id => setFiles(prev => prev.filter(f => f.id !== id))}
                                />
                            ))}
                        </div>
                    )}

                    <div className="relative mb-1">
                        <div className="max-h-96 w-full overflow-y-auto custom-scrollbar font-sans break-words transition-opacity duration-200 min-h-[2.5rem] pl-1">
                            <textarea
                                ref={textareaRef}
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                                onPaste={handlePaste}
                                onKeyDown={handleKeyDown}
                                placeholder="How can I help you today?"
                                className="w-full bg-transparent border-0 outline-none text-claude-text-100 text-[16px] placeholder:text-claude-text-400 resize-none overflow-hidden py-0 leading-relaxed block font-normal antialiased"
                                rows={1}
                                autoFocus
                                style={{ minHeight: '1.5em' }}
                            />
                        </div>
                    </div>

                    <div className="flex gap-2 w-full items-center">
                        <div className="relative flex-1 flex items-center shrink min-w-0 gap-1">
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="inline-flex items-center justify-center relative shrink-0 transition-colors duration-200 h-8 w-8 rounded-lg active:scale-95 text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200"
                                type="button"
                                aria-label="Toggle menu"
                            >
                                <Icons.Plus className="w-5 h-5" />
                            </button>

                            <div className="flex shrink min-w-8 !shrink-0">
                                <button
                                    onClick={() => setIsThinkingEnabled(!isThinkingEnabled)}
                                    className={`transition-all duration-200 h-8 w-8 flex items-center justify-center rounded-lg active:scale-95
                                        ${isThinkingEnabled
                                            ? 'text-claude-accent bg-claude-accent/10'
                                            : 'text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200'}
                                    `}
                                    aria-pressed={isThinkingEnabled}
                                    aria-label="Extended thinking"
                                >
                                    <Icons.Thinking className="w-5 h-5" />
                                </button>
                            </div>
                        </div>

                        <div className="flex flex-row items-center min-w-0 gap-1">
                            <div className="shrink-0 p-1 -m-1">
                                <ModelSelector
                                    models={models}
                                    selectedModel={selectedModel}
                                    onSelect={setSelectedModel}
                                />
                            </div>

                            <div>
                                <button
                                    onClick={handleSend}
                                    disabled={!hasContent}
                                    className={`inline-flex items-center justify-center relative shrink-0 transition-colors h-8 w-8 rounded-xl active:scale-95
                                        ${hasContent
                                            ? 'bg-claude-accent text-white hover:bg-claude-accent-hover shadow-md'
                                            : 'bg-claude-accent/30 text-white/60 cursor-default'}
                                    `}
                                    type="button"
                                    aria-label="Send message"
                                >
                                    <Icons.ArrowUp className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {isDragging && (
                <div className="absolute inset-0 bg-claude-bg-200/90 border-2 border-dashed border-claude-accent rounded-2xl z-50 flex flex-col items-center justify-center backdrop-blur-sm pointer-events-none">
                    <Icons.Archive className="w-10 h-10 text-claude-accent mb-2 animate-bounce" />
                    <p className="text-claude-accent font-medium">Drop files to upload</p>
                </div>
            )}

            <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => {
                    if (e.target.files) handleFiles(e.target.files);
                    e.target.value = '';
                }}
            />

            <div className="text-center mt-4">
                <p className="text-xs text-claude-text-500">
                    AI can make mistakes. Please check important information.
                </p>
            </div>
        </div>
    );
};

export default ClaudeChatInput;