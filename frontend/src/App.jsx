function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Fingerprint Feature Analyzer</h1>
        <p>AI-Based Fingerprint Image Feature Extraction</p>
      </header>

      <main className="main-content">
        <section className="upload-card">
          <h2>Upload Fingerprint</h2>

          <p className="upload-description">
            Upload a fingerprint image to extract measurable image features.
          </p>

          <label className="upload-area">
            <span>Choose fingerprint image</span>
            <small>JPG, JPEG or PNG</small>
            <input type="file" accept=".jpg,.jpeg,.png" />
          </label>

          <button className="analyze-button" disabled>
            Analyze Fingerprint
          </button>
        </section>
      </main>
    </div>
  )
}

export default App