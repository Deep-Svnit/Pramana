import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'PowerMind | Why We Built It',
  description: 'A separate website explaining the mission, value, and purpose behind PowerMind.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
