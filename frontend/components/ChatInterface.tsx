'use client'

import { useState, useRef, useEffect } from 'react'
import { ImagePlus, MoreVertical, Plus, Send } from 'lucide-react'
import type { StoredChatMessage } from '@/app/page'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ChatInterfaceProps {
  selectedSources: Array<{ id: string; name: string }>
  initialMessages?: StoredChatMessage[]
  onNewChat?: () => void
  onMessagesChange?: (messages: StoredChatMessage[], titleHint: string) => void
}

const getSelectionLabel = (selectedSources: Array<{ id: string; name: string }>) => {
  if (selectedSources.length === 0) return 'No files selected'
  if (selectedSources.length === 1) return selectedSources[0].name

  return `${selectedSources.length} files selected`
}

const greetingMessage: Message = {
  id: '1',
  type: 'assistant',
  content: 'Hello! I\'m PowerMind, your RAG assistant. Upload documents first, then ask me questions about them.',
  timestamp: new Date(),
}

const hydrateMessages = (messages: StoredChatMessage[]): Message[] =>
  messages.map((message) => ({
    ...message,
    timestamp: new Date(message.timestamp),
  }))

const serializeMessages = (messages: Message[]): StoredChatMessage[] =>
  messages.map((message) => ({
    ...message,
    timestamp: message.timestamp.toISOString(),
  }))

export default function ChatInterface({
  selectedSources,
  initialMessages = [],
  onNewChat = () => {},
  onMessagesChange = () => {},
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const lastSavedMessagesRef = useRef('')

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    lastSavedMessagesRef.current = JSON.stringify(initialMessages)
    setMessages(initialMessages.length > 0 ? hydrateMessages(initialMessages) : [greetingMessage])
  }, [initialMessages])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    const userMessages = messages.filter((message) => message.type === 'user')
    if (userMessages.length === 0) return

    const serializedMessages = serializeMessages(messages)
    const nextSignature = JSON.stringify(serializedMessages)
    if (nextSignature === lastSavedMessagesRef.current) return

    lastSavedMessagesRef.current = nextSignature
    onMessagesChange(serializedMessages, userMessages[0].content)
  }, [messages, onMessagesChange])

  const handleSendMessage = async () => {
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Simulate API call
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `This is a simulated response to: "${input}". In production, this would query your RAG system and retrieve relevant documents.`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1500)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="flex h-full flex-col bg-[#23262d] text-gray-100">
      <header className="flex items-center justify-between border-b border-[#313641] px-5 py-4">
        <h2 className="text-xl font-medium text-gray-100">Chat</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={onNewChat}
            className="rounded-full p-2 text-gray-300 transition hover:bg-[#2d3139] hover:text-white"
            aria-label="Start new chat"
          >
            <Plus className="h-5 w-5" />
          </button>
          <button className="rounded-full p-2 text-gray-300 transition hover:bg-[#2d3139] hover:text-white" aria-label="More options">
            <MoreVertical className="h-5 w-5" />
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="mx-auto flex max-w-4xl flex-col gap-8">
          {messages.length === 0 && (
            <div className="flex min-h-[420px] items-center justify-center">
              <div className="flex flex-col items-center gap-5 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-[#2c3038] shadow-[0_8px_20px_rgba(0,0,0,0.25)]">
                  <ImagePlus className="h-8 w-8 text-[#b9a7ff]" />
                </div>
                <div>
                  <h3 className="text-4xl font-medium tracking-tight text-gray-100">Untitled notebook</h3>
                  <p className="mt-3 text-sm text-gray-300">{getSelectionLabel(selectedSources)} · May 3, 2026</p>
                </div>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-2xl rounded-2xl px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-[#4a5dd7] text-white shadow-[0_10px_20px_rgba(74,93,215,0.25)]'
                    : 'border border-[#313641] bg-[#23272f] text-gray-100 shadow-[0_10px_20px_rgba(0,0,0,0.12)]'
                }`}
              >
                <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                <span className="mt-2 block text-xs opacity-70">{message.timestamp.toLocaleTimeString()}</span>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="rounded-2xl border border-[#313641] bg-[#23272f] px-4 py-3">
                <div className="flex gap-2">
                  <div className="h-2 w-2 animate-bounce rounded-full bg-[#b9a7ff]" />
                  <div className="h-2 w-2 animate-bounce rounded-full bg-[#b9a7ff]" style={{ animationDelay: '0.1s' }} />
                  <div className="h-2 w-2 animate-bounce rounded-full bg-[#b9a7ff]" style={{ animationDelay: '0.2s' }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t border-[#313641] bg-[#23262d] px-5 pb-3 pt-8">
        <div className="mx-auto max-w-3xl">
          <div className="flex gap-3 rounded-3xl border border-[#353a45] bg-[#1d2127] px-4 py-4 shadow-[0_12px_28px_rgba(0,0,0,0.18)]">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Start typing..."
              className="min-h-[42px] flex-1 resize-none border-0 bg-transparent px-1 py-2 text-base text-gray-100 placeholder:text-gray-500 focus:outline-none"
              rows={1}
              disabled={isLoading}
            />
            <div className="flex items-end gap-4">
              <div className="hidden items-center gap-2 text-sm text-gray-400 lg:flex">
                <span>{getSelectionLabel(selectedSources)}</span>
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
                className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-[#3c4049] text-gray-100 transition hover:bg-[#4a4f59] disabled:cursor-not-allowed disabled:bg-[#313641] disabled:text-gray-400"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>

          <p className="mt-3 text-right text-sm text-gray-400">
            Press Enter to send, Shift+Enter for new line.
          </p>
        </div>
      </div>
    </div>
  )
}
