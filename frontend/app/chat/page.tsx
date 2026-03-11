"use client";

import React, { useState } from "react";
import ClaudeChatInput from "@/components/ui/claude-style-chat-input";
import type { AttachedFile } from "@/components/ui/claude-style-chat-input";

export default function ChatPage() {
    const [messages, setMessages] = useState<string[]>([]);

    const handleSendMessage = (data: {
        message: string;
        files: AttachedFile[];
        model: string;
        isThinkingEnabled: boolean;
    }) => {
        console.log("Sending message:", data.message);
        console.log("Model:", data.model);
        console.log("Thinking:", data.isThinkingEnabled);
        console.log("Attached files:", data.files);
        setMessages(prev => [...prev, data.message]);
    };

    const currentHour = new Date().getHours();
    let greeting = "Good morning";
    if (currentHour >= 12 && currentHour < 18) {
        greeting = "Good afternoon";
    } else if (currentHour >= 18) {
        greeting = "Good evening";
    }

    const userName = "Rajat";

    return (
        <div className="min-h-screen w-full bg-claude-bg-0 flex flex-col items-center justify-center p-4 font-sans text-claude-text-100 transition-colors duration-200">
            <div className="w-full max-w-3xl mb-8 sm:mb-12 text-center animate-claude-fade-in">
                <div className="w-24 h-24 mx-auto mb-6 flex items-center justify-center">
                    <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                        <defs>
                            <ellipse id="hero-petal" cx="100" cy="100" rx="90" ry="22" />
                        </defs>
                        <g fill="#D46B4F" fillRule="evenodd">
                            <use href="#hero-petal" transform="rotate(0 100 100)" />
                            <use href="#hero-petal" transform="rotate(45 100 100)" />
                            <use href="#hero-petal" transform="rotate(90 100 100)" />
                            <use href="#hero-petal" transform="rotate(135 100 100)" />
                        </g>
                    </svg>
                </div>
                <h1 className="text-3xl sm:text-4xl font-serif font-light text-claude-text-200 mb-3 tracking-tight">
                    {greeting},{" "}
                    <span className="relative inline-block pb-2">
                        {userName}
                        <svg
                            className="absolute w-[140%] h-[20px] -bottom-1 -left-[5%] text-claude-accent"
                            viewBox="0 0 140 24"
                            fill="none"
                            preserveAspectRatio="none"
                            aria-hidden="true"
                        >
                            <path
                                d="M6 16 Q 70 24, 134 14"
                                stroke="currentColor"
                                strokeWidth="3"
                                strokeLinecap="round"
                                fill="none"
                            />
                        </svg>
                    </span>
                </h1>
            </div>

            <ClaudeChatInput onSendMessage={handleSendMessage} />

            <div className="flex flex-wrap justify-center gap-2 mt-4 max-w-2xl mx-auto px-4">
                <button className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-claude-text-300 bg-transparent border border-claude-bg-300 dark:border-claude-bg-300/50 rounded-full hover:bg-claude-bg-200 hover:text-claude-text-200 transition-colors duration-150">
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                    </svg>
                    Write
                </button>
                <button className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-claude-text-300 bg-transparent border border-claude-bg-300 dark:border-claude-bg-300/50 rounded-full hover:bg-claude-bg-200 hover:text-claude-text-200 transition-colors duration-150">
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M22 10v6M2 10l10-5 10 5-10 5z" /><path d="M6 12v5c0 2.5 6 2.5 6 2.5s6 0 6-2.5v-5" />
                    </svg>
                    Learn
                </button>
                <button className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-claude-text-300 bg-transparent border border-claude-bg-300 dark:border-claude-bg-300/50 rounded-full hover:bg-claude-bg-200 hover:text-claude-text-200 transition-colors duration-150">
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
                    </svg>
                    Code
                </button>
                <button className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-claude-text-300 bg-transparent border border-claude-bg-300 dark:border-claude-bg-300/50 rounded-full hover:bg-claude-bg-200 hover:text-claude-text-200 transition-colors duration-150">
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" />
                    </svg>
                    Life stuff
                </button>
            </div>

            {messages.length > 0 && (
                <div className="w-full max-w-2xl mt-8 space-y-3">
                    {messages.map((msg, i) => (
                        <div
                            key={i}
                            className="px-4 py-3 bg-claude-bg-100 border border-claude-bg-300 rounded-xl text-sm text-claude-text-200 animate-claude-fade-in"
                        >
                            {msg}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}