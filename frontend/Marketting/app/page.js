export default function Home() {
  return (
    <main>
      {/* Hero Section */}
      <section className="hero">
        <span className="badge">Medhavi x PowerMind</span>
        <h1>Transform documents into intelligent answers</h1>
        <p className="lead">
          PowerMind is a retrieval-augmented generation chatbot built by Medhavi to help teams find answers in their documents instantly. 
          Upload PDFs, earnings reports, research papers, or any document—then ask natural questions and get accurate, sourced answers.
        </p>

        <div className="ctaRow">
          <a className="btn btnPrimary" href="http://localhost:3000" target="_blank" rel="noreferrer">
            Try PowerMind Now
          </a>
          <a className="btn btnSecondary" href="#how-it-works">
            Learn How It Works
          </a>
        </div>
      </section>

      {/* Why Medhavi Built PowerMind */}
      <section className="why">
        <span className="sectionLabel">Our Vision</span>
        <h2>Why Medhavi built PowerMind</h2>
        <p className="sectionLead">
          Teams spend countless hours searching through documents for critical information. This inefficiency delays decisions, 
          fragments knowledge, and wastes valuable time that could go toward strategy and growth.
        </p>

        <div className="gridThree">
          <article className="card">
            <div className="cardIcon">📄</div>
            <h3>The problem we saw</h3>
            <p>
              Financial teams manually scan earnings reports. Investment committees dig through research. 
              Valuable insights stay buried. Context switching across tools breaks focus and kills productivity.
            </p>
          </article>

          <article className="card">
            <div className="cardIcon">⚡</div>
            <h3>The solution we built</h3>
            <p>
              PowerMind combines advanced retrieval and AI-powered generation to answer questions from your documents. 
              No manual searching. No context switching. Just clear, sourced answers in seconds.
            </p>
          </article>

          <article className="card">
            <div className="cardIcon">🎯</div>
            <h3>The impact</h3>
            <p>
              Faster decisions. Better collaboration. Reduced cognitive load. Teams that use PowerMind reclaim hours per week 
              and make decisions grounded in actual data.
            </p>
          </article>
        </div>
      </section>

      {/* Key Features */}
      <section className="features">
        <span className="sectionLabel">Capabilities</span>
        <h2>Powerful features for document intelligence</h2>

        <div className="featureGrid">
          <div className="featureCard">
            <h3>📤 Smart Document Upload</h3>
            <p>Drag and drop or select multiple PDFs, reports, and documents. PowerMind handles files up to several megabytes with ease.</p>
          </div>

          <div className="featureCard">
            <h3>💬 Natural Language Questions</h3>
            <p>Ask questions in plain English. PowerMind understands context and retrieves relevant information from your documents automatically.</p>
          </div>

          <div className="featureCard">
            <h3>🔗 Sourced Answers</h3>
            <p>Every answer is grounded in your actual documents. Know exactly where information comes from and trace it back to the source.</p>
          </div>

          <div className="featureCard">
            <h3>📊 Multi-Document Context</h3>
            <p>Analyze patterns across multiple documents. Compare earnings reports, research papers, or financial statements side by side.</p>
          </div>

          <div className="featureCard">
            <h3>🔍 Full-Text Search</h3>
            <p>Find documents and content quickly. Select which documents to focus on for each conversation.</p>
          </div>

          <div className="featureCard">
            <h3>🛡️ Privacy-First Design</h3>
            <p>Your documents stay in your control. PowerMind processes your data securely without exposing sensitive information.</p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="howItWorks">
        <span className="sectionLabel">The Process</span>
        <h2>How PowerMind works</h2>

        <div className="stepsContainer">
          <div className="step">
            <div className="stepNumber">1</div>
            <h3>Upload Documents</h3>
            <p>Start by uploading the documents you want to analyze. PDF, reports, presentations—PowerMind handles them all.</p>
          </div>

          <div className="stepDivider">→</div>

          <div className="step">
            <div className="stepNumber">2</div>
            <h3>Ask Your Question</h3>
            <p>Type a natural language question about your documents. PowerMind understands context and intent.</p>
          </div>

          <div className="stepDivider">→</div>

          <div className="step">
            <div className="stepNumber">3</div>
            <h3>Get Sourced Answers</h3>
            <p>Receive instant answers backed by quotes from your documents. Know exactly where the information comes from.</p>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="useCases">
        <span className="sectionLabel">Real-World Applications</span>
        <h2>Who uses PowerMind</h2>

        <div className="gridTwo">
          <div className="caseCard">
            <h3>📈 Financial Analysts</h3>
            <p>
              Extract key metrics from earnings reports, SEC filings, and financial statements in seconds. 
              Compare company performance across quarters and years without manual data entry.
            </p>
          </div>

          <div className="caseCard">
            <h3>💼 Investment Teams</h3>
            <p>
              Research opportunities faster by querying research reports, market analyses, and due diligence documents. 
              Find patterns and insights that inform better investment decisions.
            </p>
          </div>

          <div className="caseCard">
            <h3>📚 Legal & Compliance</h3>
            <p>
              Navigate complex regulations and contracts. Ask about specific clauses, obligations, and compliance requirements 
              without reading hundreds of pages.
            </p>
          </div>

          <div className="caseCard">
            <h3>🏥 Research & Academia</h3>
            <p>
              Synthesize information across multiple research papers and datasets. Find relevant citations and methodologies quickly.
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="ctaFinal">
        <h2>Ready to transform how your team works with documents?</h2>
        <p>PowerMind is built to save you time, reduce errors, and make document intelligence accessible to everyone.</p>
        
        <div className="ctaRow">
          <a className="btn btnPrimary" href="http://localhost:3000" target="_blank" rel="noreferrer">
            Try PowerMind Free
          </a>
          <a className="btn btnSecondary" href="#" aria-disabled="true">
            Schedule a Demo (coming soon)
          </a>
        </div>
      </section>

      {/* Footer Note */}
      <section className="footer">
        <p className="footerNote">Built with ❤️ by Medhavi | PowerMind runs on port 3000, this site on port 3001</p>
      </section>
    </main>
  )
}
