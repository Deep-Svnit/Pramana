'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import ChatInterface from '@/components/ChatInterface'
import SourcesPanel from '@/components/SourcesPanel'

export interface UploadedSource {
  id: string
  name: string
  size: number
  status: 'completed' | 'uploading' | 'failed'
  uploadedAt: string
  previewUrl?: string
}

export interface StoredChatMessage {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: string
}

export interface ChatHistoryItem {
  id: string
  title: string
  preview: string
  createdAt: string
  updatedAt: string
  messages: StoredChatMessage[]
}

const defaultSources: UploadedSource[] = [
  {
    id: 'default-ael-earnings-presentation',
    name: 'AEL_Earnings_Presentation_Q2-FY26_copy.pdf',
    size: 2523226,
    status: 'completed',
    uploadedAt: 'Default document',
    previewUrl: '/api/preview/default-ael-earnings-presentation',
  },
]

const CHAT_HISTORY_STORAGE_KEY = 'powermind-chat-history'
const emptyChatMessages: StoredChatMessage[] = []

export default function Home() {
  const [sources, setSources] = useState<UploadedSource[]>(defaultSources)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([defaultSources[0].id])
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [chatDraftKey, setChatDraftKey] = useState(0)
  const [sourcesWidth, setSourcesWidth] = useState(380)
  const [isResizingSources, setIsResizingSources] = useState(false)
  const [previewSource, setPreviewSource] = useState<UploadedSource | null>(null)
  const [previewWidth, setPreviewWidth] = useState(480)
  const [isResizingPreview, setIsResizingPreview] = useState(false)

  const selectedSources = useMemo(
    () => sources.filter((source) => selectedSourceIds.includes(source.id)),
    [selectedSourceIds, sources]
  )

  const filteredSources = useMemo(() => {
    const term = searchTerm.trim().toLowerCase()
    if (!term) return sources

    return sources.filter((source) => source.name.toLowerCase().includes(term))
  }, [searchTerm, sources]
)

  const activeChat = useMemo(
    () => chatHistory.find((chat) => chat.id === activeChatId) ?? null,
    [activeChatId, chatHistory]
  )

  useEffect(() => {
    try {
      const storedHistory = window.localStorage.getItem(CHAT_HISTORY_STORAGE_KEY)
      if (!storedHistory) return

      const parsedHistory = JSON.parse(storedHistory) as ChatHistoryItem[]
      setChatHistory(parsedHistory)
      setActiveChatId(parsedHistory[0]?.id ?? null)
    } catch {
      setChatHistory([])
    }
  }, [])

  useEffect(() => {
    window.localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(chatHistory))
  }, [chatHistory])

  const handleChatMessagesChange = useCallback((messages: StoredChatMessage[], titleHint: string) => {
    const userMessages = messages.filter((message) => message.type === 'user')
    if (userMessages.length === 0) return

    const now = new Date().toISOString()
    const nextChatId = activeChatId ?? `chat-${Date.now()}`
    const title = titleHint.trim().slice(0, 60) || 'New chat'
    const preview = [...messages].reverse().find((message) => message.type === 'assistant')?.content || title

    setChatHistory((currentHistory) => {
      const nextItem: ChatHistoryItem = {
        id: nextChatId,
        title,
        preview: preview.slice(0, 90),
        createdAt: currentHistory.find((chat) => chat.id === nextChatId)?.createdAt ?? now,
        updatedAt: now,
        messages,
      }

      const withoutCurrent = currentHistory.filter((chat) => chat.id !== nextChatId)
      return [nextItem, ...withoutCurrent]
    })

    if (!activeChatId) {
      setActiveChatId(nextChatId)
    }
  }, [activeChatId])

  const startNewChat = () => {
    setActiveChatId(null)
    setChatDraftKey((currentKey) => currentKey + 1)
  }

  const selectChat = (chatId: string) => {
    setActiveChatId(chatId)
    setChatDraftKey((currentKey) => currentKey + 1)
  }

  const deleteChat = (chatId: string) => {
    setChatHistory((currentHistory) => currentHistory.filter((chat) => chat.id !== chatId))
    if (activeChatId === chatId) {
      startNewChat()
    }
  }

  useEffect(() => {
    if (!isResizingSources) return

    const handleMouseMove = (event: MouseEvent) => {
      setSourcesWidth(Math.min(620, Math.max(300, event.clientX)))
    }

    const handleMouseUp = () => {
      setIsResizingSources(false)
    }

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizingSources])

  useEffect(() => {
    if (!isResizingPreview) return

    const handleMouseMove = (event: MouseEvent) => {
      const nextWidth = window.innerWidth - event.clientX
      setPreviewWidth(Math.min(760, Math.max(360, nextWidth)))
    }

    const handleMouseUp = () => {
      setIsResizingPreview(false)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizingPreview])

  useEffect(() => {
    if (previewSource) {
      setPreviewWidth((currentWidth) => Math.min(760, Math.max(360, currentWidth)))
    }
  }, [previewSource])

  return (
    <div className="flex h-screen overflow-hidden bg-[#1f2229] text-gray-100">
      <SourcesPanel
        sources={filteredSources}
        allSources={sources}
        onSourcesChange={setSources}
        searchTerm={searchTerm}
        onSearchTermChange={setSearchTerm}
        onSelectedSourceIdsChange={setSelectedSourceIds}
        onPreviewSourceChange={setPreviewSource}
        chatHistory={chatHistory}
        activeChatId={activeChatId}
        onNewChat={startNewChat}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
        width={sourcesWidth}
        onResizeStart={() => setIsResizingSources(true)}
      />
      <div className="flex min-w-0 flex-1">
        <div className="min-w-0 flex-1">
          <ChatInterface
            key={chatDraftKey}
            selectedSources={selectedSources}
            initialMessages={activeChat?.messages ?? emptyChatMessages}
            onNewChat={startNewChat}
            onMessagesChange={handleChatMessagesChange}
          />
        </div>

        {previewSource && (
          <aside
            className="relative flex h-full flex-col border-l border-[#2e323b] bg-[#181b21] shadow-[-1px_0_0_rgba(255,255,255,0.04)]"
            style={{ width: `${previewWidth}px` }}
          >
            <button
              type="button"
              aria-label="Resize preview pane"
              onMouseDown={() => setIsResizingPreview(true)}
              className="absolute left-0 top-0 h-full w-1 cursor-col-resize bg-transparent transition-colors hover:bg-[#5a67ff]"
            />

            <div className="flex items-center justify-between border-b border-[#313641] px-5 py-4 pl-6">
              <div className="min-w-0 pr-4">
                <p className="truncate text-base font-semibold text-gray-100">PDF Preview</p>
                <p className="truncate text-sm text-gray-400">{previewSource.name}</p>
              </div>
              <button
                onClick={() => setPreviewSource(null)}
                className="inline-flex shrink-0 items-center gap-2 rounded-full border border-[#3a3f49] bg-[#2a2f37] px-4 py-2 text-sm text-gray-100 transition hover:bg-[#333944]"
              >
                Close
              </button>
            </div>

            <div className="flex-1 bg-[#111319] p-4 pl-6">
              <iframe
                title={previewSource.name}
                src={previewSource.previewUrl ?? `/api/preview/${previewSource.id}`}
                className="h-full w-full rounded-[22px] border border-[#313641] bg-white"
              />
            </div>
          </aside>
        )}
      </div>
    </div>
  )
}
