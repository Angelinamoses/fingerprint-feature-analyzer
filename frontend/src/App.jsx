import { useState } from "react"

const API_URL = "http://127.0.0.1:8000"

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [imageDimensions, setImageDimensions] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]

    setError(null)
    setResults(null)

    if (!file) {
      setSelectedFile(null)
      setPreviewUrl(null)
      setImageDimensions(null)
      return
    }

    const allowedTypes = [
      "image/jpeg",
      "image/png",
    ]

    if (!allowedTypes.includes(file.type)) {
      setError("Please upload a JPG, JPEG, or PNG image.")
      setSelectedFile(null)
      setPreviewUrl(null)
      setImageDimensions(null)
      return
    }

    setSelectedFile(file)

    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)

    const image = new Image()

    image.onload = () => {
      setImageDimensions({
        width: image.width,
        height: image.height,
      })

      URL.revokeObjectURL(objectUrl)
    }

    image.src = objectUrl
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      const response = await fetch(
        `${API_URL}/analyze-fingerprint`,
        {
          method: "POST",
          body: formData,
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to analyze fingerprint image."
        )
      }

      if (!data.success) {
        throw new Error(
          data.error || "Unable to analyze fingerprint image."
        )
      }

      setResults(data.features)
    } catch (err) {
      setError(
        err.message ||
        "Unable to connect to the fingerprint analysis server."
      )
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setImageDimensions(null)
    setResults(null)
    setError(null)
  }

  const formatValue = (value, suffix = "") => {
    if (value === null || value === undefined) {
      return "Not detected"
    }

    if (typeof value === "number") {
      return `${value.toFixed(3)}${suffix}`
    }

    return `${value}${suffix}`
  }

  const downloadJSON = () => {
    if (!results) {
      return
    }

    const report = {
      filename: selectedFile?.name || null,
      image_dimensions: imageDimensions,
      features: results,
    }

    const blob = new Blob(
      [JSON.stringify(report, null, 2)],
      {
        type: "application/json",
      }
    )

    const url = URL.createObjectURL(blob)

    const link = document.createElement("a")
    link.href = url
    link.download = "fingerprint-analysis-report.json"

    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    URL.revokeObjectURL(url)
  }

  return (
    <div className="app">

      <header className="header">
        <h1>Fingerprint Feature Analyzer</h1>

        <p>
          AI-Based Fingerprint Image Feature Extraction
        </p>
      </header>


      <main className="main-content">

        {/* Upload Section */}

        <section className="upload-card">

          <h2>Upload Fingerprint</h2>

          <p className="upload-description">
            Upload a fingerprint image to extract measurable
            image features.
          </p>


          <label className="upload-area">

            <span>
              Choose fingerprint image
            </span>

            <small>
              JPG, JPEG or PNG
            </small>

            <input
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              onChange={handleFileChange}
            />

          </label>


          {selectedFile && (
            <div className="file-info">

              <strong>
                {selectedFile.name}
              </strong>

              {imageDimensions && (
                <span>
                  {imageDimensions.width} ×{" "}
                  {imageDimensions.height}px
                </span>
              )}

            </div>
          )}


          {previewUrl && (
            <div className="preview-section">

              <h3>Image Preview</h3>

              <img
                src={previewUrl}
                alt="Fingerprint preview"
                className="fingerprint-preview"
              />

            </div>
          )}


          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
          >
            {loading
              ? "Analyzing fingerprint..."
              : "Analyze Fingerprint"}
          </button>


          {selectedFile && !loading && (
            <button
              className="reset-button"
              onClick={handleReset}
            >
              Reset
            </button>
          )}

        </section>


        {/* Error */}

        {error && (
          <section className="error-card">
            <strong>Analysis Error</strong>

            <p>{error}</p>
          </section>
        )}


        {/* Results */}

        {results && (
          <section className="results-section">

            <div className="results-header">

              <div>
                <h2>Analysis Results</h2>

                <p>
                  Extracted fingerprint image features
                </p>
              </div>

              <button
                className="download-button"
                onClick={downloadJSON}
              >
                Download JSON
              </button>

            </div>


            {/* Pattern */}

            <div className="result-card">

              <h3>Fingerprint Pattern</h3>

              <div className="single-result">

                <span>Pattern type</span>

                <strong>
                  {results.pattern_type || "Not detected"}
                </strong>

              </div>

            </div>


            {/* Image Quality */}

            <div className="result-card">

              <h3>Image Quality</h3>

              <div className="feature-grid">

                <div className="feature-item">
                  <span>Mean intensity</span>
                  <strong>
                    {formatValue(
                      results.image_quality?.mean_intensity
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Intensity standard deviation</span>
                  <strong>
                    {formatValue(
                      results.image_quality?.intensity_std
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Local variance</span>
                  <strong>
                    {formatValue(
                      results.image_quality?.local_variance
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Foreground pixels</span>
                  <strong>
                    {results.image_quality?.foreground_pixels ??
                      "Not detected"}
                  </strong>
                </div>

              </div>

            </div>


            {/* Core / Delta */}

            <div className="result-card">

              <h3>Core & Delta</h3>

              <div className="feature-grid">

                <div className="feature-item">
                  <span>Core X</span>
                  <strong>
                    {results.core?.x ?? "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Core Y</span>
                  <strong>
                    {results.core?.y ?? "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Core confidence</span>
                  <strong>
                    {results.core?.confidence
                      ? formatValue(results.core.confidence)
                      : "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Delta X</span>
                  <strong>
                    {results.delta?.x ?? "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Delta Y</span>
                  <strong>
                    {results.delta?.y ?? "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Delta confidence</span>
                  <strong>
                    {results.delta?.confidence
                      ? formatValue(results.delta.confidence)
                      : "Not detected"}
                  </strong>
                </div>

              </div>

            </div>


            {/* Ridge Features */}

            <div className="result-card">

              <h3>Ridge Features</h3>

              <div className="feature-grid">

                <div className="feature-item">
                  <span>Ridge density</span>
                  <strong>
                    {formatValue(
                      results.ridge?.density
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Dominant orientation</span>
                  <strong>
                    {formatValue(
                      results.ridge?.orientation_degrees,
                      "°"
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Orientation coherence</span>
                  <strong>
                    {formatValue(
                      results.ridge?.orientation_coherence
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Ridge frequency</span>
                  <strong>
                    {formatValue(
                      results.ridge?.frequency
                    )}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Ridge spacing</span>
                  <strong>
                    {formatValue(
                      results.ridge?.spacing_pixels,
                      " px"
                    )}
                  </strong>
                </div>

              </div>

            </div>


            {/* Minutiae */}

            <div className="result-card">

              <h3>Minutiae</h3>

              <div className="feature-grid">

                <div className="feature-item">
                  <span>Total minutiae</span>
                  <strong>
                    {results.minutiae?.total ??
                      "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Ridge endings</span>
                  <strong>
                    {results.minutiae?.ridge_endings ??
                      "Not detected"}
                  </strong>
                </div>

                <div className="feature-item">
                  <span>Bifurcations</span>
                  <strong>
                    {results.minutiae?.bifurcations ??
                      "Not detected"}
                  </strong>
                </div>

              </div>

              <p className="not-detected-note">
                Minutiae points are currently not reported
                because the experimental detector has not
                reached a reliable validation level.
              </p>

            </div>


            {/* Transparency */}

            <div className="transparency-note">

              <strong>
                Scientific transparency
              </strong>

              <p>
                Features are generated by automated image
                analysis. Confidence values represent model
                or measurement estimates and should not be
                interpreted as certainty.
              </p>

              <p>
                This application performs image feature
                extraction only. It does not identify a
                person, compare fingerprints against a
                database, or make forensic authentication
                decisions.
              </p>

            </div>

          </section>
        )}

      </main>

    </div>
  )
}

export default App