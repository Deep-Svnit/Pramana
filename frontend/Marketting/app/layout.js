import './globals.css'

export const metadata = {
  title: 'Medhavi | PowerMind Vision',
  description: 'Why Medhavi built PowerMind and how it improves document intelligence workflows.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
