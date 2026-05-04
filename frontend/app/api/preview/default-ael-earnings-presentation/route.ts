import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import path from 'path'

export async function GET() {
  const filePath = path.join(process.cwd(), 'AEL_Earnings_Presentation_Q2-FY26_copy.pdf')
  const fileBuffer = await readFile(filePath)

  return new NextResponse(fileBuffer, {
    headers: {
      'Content-Type': 'application/pdf',
      'Content-Disposition': 'inline; filename="AEL_Earnings_Presentation_Q2-FY26_copy.pdf"',
    },
  })
}
