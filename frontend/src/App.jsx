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

    const allowedTypes = ["image/jpeg", "image/png"]

    if (!allowedTypes.includes(file.type)) {
      setError("Please upload a JPG, JPEG, or PNG image.")
      setSelectedFile(null)
      setPreviewUrl(null)
      setImageDimensions(null)
      return
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    const objectUrl = URL.createObjectURL(file)

    setSelectedFile(file)
    setPreviewUrl(objectUrl)

    const image = new Image()

    image.onload = () => {
      setImageDimensions({
        width: image.width,
        height: image.height,
      })
    }

    image.src = objectUrl
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

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
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setSelectedFile(null)
    setPreviewUrl(null)
    setImageDimensions(null)
    setResults(null)
    setError(null)
  }

  const formatNumber = (value, decimals = 3) => {
    if (value === null || value === undefined) {
      return "Not detected"
    }

    if (typeof value !== "number") {
      return value
    }

    return value.toFixed(decimals)
  }

  const displayValue = (value) => {
    if (value === null || value === undefined || value === "") {
      return "Not detected"
    }

    return value
  }

  const downloadJSON = () => {
    if (!results) return

    const report = {
      filename: selectedFile?.name || null,
      image_dimensions: imageDimensions,
      features: results,
    }

    const blob = new Blob(
      [JSON.stringify(report, null, 2)],
      { type: "application/json" }
    )

    const url = URL.createObjectURL(blob)

    const link = document.createElement("a")
    link.href = url
    link.download = "fingerprint-analysis-report.json"

    document.body.appendChild(link)
    link.click()
    link.remove()

    URL.revokeObjectURL(url)
  }

  const downloadCSV = () => {
    if (!results) return

    const rows = [
      ["Category", "Feature", "Value", "Status"],

      [
        "Pattern",
        "Pattern Type",
        displayValue(results.pattern_type),
        results.pattern_type ? "Detected" : "Not detected",
      ],

      [
        "Image Quality",
        "Mean Intensity",
        formatNumber(
          results.image_quality?.mean_intensity
        ),
        "Measured",
      ],

      [
        "Image Quality",
        "Intensity Standard Deviation",
        formatNumber(
          results.image_quality?.intensity_std
        ),
        "Measured",
      ],

      [
        "Image Quality",
        "Local Variance",
        formatNumber(
          results.image_quality?.local_variance
        ),
        "Measured",
      ],

      [
        "Image Quality",
        "Foreground Pixels",
        displayValue(
          results.image_quality?.foreground_pixels
        ),
        "Measured",
      ],

      [
        "Core",
        "X",
        displayValue(results.core?.x),
        results.core?.x !== null ? "Detected" : "Not detected",
      ],

      [
        "Core",
        "Y",
        displayValue(results.core?.y),
        results.core?.y !== null ? "Detected" : "Not detected",
      ],

      [
        "Core",
        "Confidence",
        displayValue(results.core?.confidence),
        results.core?.confidence !== null
          ? "Available"
          : "Not available",
      ],

      [
        "Delta",
        "X",
        displayValue(results.delta?.x),
        results.delta?.x !== null ? "Detected" : "Not detected",
      ],

      [
        "Delta",
        "Y",
        displayValue(results.delta?.y),
        results.delta?.y !== null ? "Detected" : "Not detected",
      ],

      [
        "Delta",
        "Confidence",
        displayValue(results.delta?.confidence),
        results.delta?.confidence !== null
          ? "Available"
          : "Not available",
      ],

      [
        "Ridge",
        "Density",
        formatNumber(results.ridge?.density),
        "Measured",
      ],

      [
        "Ridge",
        "Dominant Orientation",
        results.ridge?.orientation_degrees !== null &&
        results.ridge?.orientation_degrees !== undefined
          ? `${formatNumber(
              results.ridge.orientation_degrees,
              2
            )}°`
          : "Not detected",
        "Measured",
      ],

      [
        "Ridge",
        "Orientation Coherence",
        formatNumber(
          results.ridge?.orientation_coherence
        ),
        "Measured",
      ],

      [
        "Ridge",
        "Frequency",
        formatNumber(results.ridge?.frequency),
        "Measured",
      ],

      [
        "Ridge",
        "Spacing",
        results.ridge?.spacing_pixels !== null &&
        results.ridge?.spacing_pixels !== undefined
          ? `${formatNumber(
              results.ridge.spacing_pixels,
              2
            )} px`
          : "Not detected",
        "Measured",
      ],

      [
        "Minutiae",
        "Total",
        displayValue(results.minutiae?.total),
        results.minutiae?.total !== null
          ? "Detected"
          : "Not detected",
      ],

      [
        "Minutiae",
        "Ridge Endings",
        displayValue(results.minutiae?.ridge_endings),
        results.minutiae?.ridge_endings !== null
          ? "Detected"
          : "Not detected",
      ],

      [
        "Minutiae",
        "Bifurcations",
        displayValue(results.minutiae?.bifurcations),
        results.minutiae?.bifurcations !== null
          ? "Detected"
          : "Not detected",
      ],
    ]

    const csv = rows
      .map((row) =>
        row
          .map((cell) => `"${String(cell).replaceAll('"', '""')}"`)
          .join(",")
      )
      .join("\n")

    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    })

    const url = URL.createObjectURL(blob)

    const link = document.createElement("a")
    link.href = url
    link.download = "fingerprint-analysis-report.csv"

    document.body.appendChild(link)
    link.click()
    link.remove()

    URL.revokeObjectURL(url)
  }

  return (
    <div className="app">

      <header className="header">

        <div className="header-content">

          <div className="brand-mark">
            FP
          </div>

          <div>
            <h1>Fingerprint Feature Analyzer</h1>

            <p>
              AI-Based Fingerprint Image Feature Extraction
            </p>
          </div>

        </div>

      </header>


      <main className="main-content">

        {/* Upload */}

        <section className="card upload-card">

          <div className="section-heading">

            <div>
              <span className="eyebrow">
                IMAGE INPUT
              </span>

              <h2>
                Upload a fingerprint
              </h2>

              <p>
                Upload a grayscale or color fingerprint image
                for automated feature extraction.
              </p>
            </div>

          </div>


          <label className="upload-area">

            <div className="upload-icon">
              ↑
            </div>

            <strong>
              Choose fingerprint image
            </strong>

            <span>
              JPG, JPEG or PNG
            </span>

            <input
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              onChange={handleFileChange}
            />

          </label>


          {selectedFile && (
            <div className="file-summary">

              <div>
                <span className="file-label">
                  Selected file
                </span>

                <strong>
                  {selectedFile.name}
                </strong>
              </div>

              {imageDimensions && (
                <div>
                  <span className="file-label">
                    Dimensions
                  </span>

                  <strong>
                    {imageDimensions.width} ×{" "}
                    {imageDimensions.height}px
                  </strong>
                </div>
              )}

            </div>
          )}


          {previewUrl && (
            <div className="preview-wrapper">

              <div className="preview-heading">
                <h3>Image Preview</h3>

                <span>
                  Original image
                </span>
              </div>

              <div className="preview-container">

                <img
                  src={previewUrl}
                  alt="Uploaded fingerprint"
                  className="fingerprint-preview"
                />

              </div>

            </div>
          )}


          <div className="action-row">

            <button
              className="primary-button"
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
            >
              {loading
                ? "Analyzing fingerprint..."
                : "Analyze Fingerprint"}
            </button>

            {selectedFile && !loading && (
              <button
                className="secondary-button"
                onClick={handleReset}
              >
                Reset
              </button>
            )}

          </div>


          {loading && (
            <div className="loading-state">

              <div className="spinner" />

              <div>
                <strong>
                  Analyzing fingerprint...
                </strong>

                <span>
                  Running preprocessing and feature extraction
                </span>
              </div>

            </div>
          )}

        </section>


        {/* Error */}

        {error && (
          <section className="error-card">

            <div className="error-icon">
              !
            </div>

            <div>
              <strong>
                Analysis failed
              </strong>

              <p>
                {error}
              </p>
            </div>

          </section>
        )}


        {/* Results */}

        {results && (
          <section className="results-section">

            <div className="results-header">

              <div>
                <span className="eyebrow">
                  ANALYSIS COMPLETE
                </span>

                <h2>
                  Extracted Features
                </h2>

                <p>
                  Measurements generated from the uploaded
                  fingerprint image.
                </p>
              </div>

              <div className="download-actions">

                <button
                  className="secondary-button"
                  onClick={downloadCSV}
                >
                  Download CSV
                </button>

                <button
                  className="secondary-button"
                  onClick={downloadJSON}
                >
                  Download JSON
                </button>

              </div>

            </div>


            {/* Pattern */}

            <div className="card result-card">

              <div className="card-title-row">

                <div>
                  <h3>Pattern</h3>

                  <p>
                    Global fingerprint pattern classification
                  </p>
                </div>

                <span
                  className={
                    results.pattern_type
                      ? "status-badge detected"
                      : "status-badge unavailable"
                  }
                >
                  {results.pattern_type
                    ? "Detected"
                    : "Not detected"}
                </span>

              </div>


              <div className="feature-table">

                <div className="table-row table-header">
                  <span>Feature</span>
                  <span>Value</span>
                  <span>Status</span>
                </div>

                <div className="table-row">
                  <span>Pattern type</span>

                  <strong>
                    {displayValue(
                      results.pattern_type
                    )}
                  </strong>

                  <span>
                    {results.pattern_type
                      ? "Detected"
                      : "Not detected"}
                  </span>
                </div>

              </div>

            </div>


            {/* Core / Delta */}

            <div className="card result-card">

              <div className="card-title-row">

                <div>
                  <h3>Core & Delta</h3>

                  <p>
                    Singular-point candidates from ridge orientation
                    analysis
                  </p>
                </div>

              </div>


              <div className="feature-table">

                <div className="table-row table-header">
                  <span>Feature</span>
                  <span>Value</span>
                  <span>Status</span>
                </div>

                <div className="table-row">
                  <span>Core X</span>
                  <strong>
                    {displayValue(results.core?.x)}
                  </strong>
                  <span>
                    {results.core?.x !== null
                      ? "Detected"
                      : "Not detected"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Core Y</span>
                  <strong>
                    {displayValue(results.core?.y)}
                  </strong>
                  <span>
                    {results.core?.y !== null
                      ? "Detected"
                      : "Not detected"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Core confidence</span>
                  <strong>
                    {results.core?.confidence !== null
                      ? formatNumber(results.core.confidence)
                      : "Not detected"}
                  </strong>
                  <span>
                    {results.core?.confidence !== null
                      ? "Available"
                      : "Not available"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Delta X</span>
                  <strong>
                    {displayValue(results.delta?.x)}
                  </strong>
                  <span>
                    {results.delta?.x !== null
                      ? "Detected"
                      : "Not detected"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Delta Y</span>
                  <strong>
                    {displayValue(results.delta?.y)}
                  </strong>
                  <span>
                    {results.delta?.y !== null
                      ? "Detected"
                      : "Not detected"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Delta confidence</span>
                  <strong>
                    {results.delta?.confidence !== null
                      ? formatNumber(results.delta.confidence)
                      : "Not detected"}
                  </strong>
                  <span>
                    {results.delta?.confidence !== null
                      ? "Available"
                      : "Not available"}
                  </span>
                </div>

              </div>

            </div>


            {/* Ridge */}

            <div className="card result-card">

              <div className="card-title-row">

                <div>
                  <h3>Ridge Features</h3>

                  <p>
                    Measured properties of the fingerprint ridge
                    structure
                  </p>
                </div>

                <span className="status-badge detected">
                  Measured
                </span>

              </div>


              <div className="feature-table">

                <div className="table-row table-header">
                  <span>Feature</span>
                  <span>Value</span>
                  <span>Unit / Status</span>
                </div>

                <div className="table-row">
                  <span>Ridge density</span>
                  <strong>
                    {formatNumber(results.ridge?.density)}
                  </strong>
                  <span>
                    peaks / pixel
                  </span>
                </div>

                <div className="table-row">
                  <span>Dominant ridge orientation</span>
                  <strong>
                    {results.ridge?.orientation_degrees !== null &&
                    results.ridge?.orientation_degrees !== undefined
                      ? `${formatNumber(
                          results.ridge.orientation_degrees,
                          2
                        )}°`
                      : "Not detected"}
                  </strong>
                  <span>
                    degrees
                  </span>
                </div>

                <div className="table-row">
                  <span>Orientation coherence</span>
                  <strong>
                    {formatNumber(
                      results.ridge?.orientation_coherence
                    )}
                  </strong>
                  <span>
                    0–1
                  </span>
                </div>

                <div className="table-row">
                  <span>Ridge frequency</span>
                  <strong>
                    {formatNumber(
                      results.ridge?.frequency
                    )}
                  </strong>
                  <span>
                    cycles / pixel
                  </span>
                </div>

                <div className="table-row">
                  <span>Ridge spacing</span>
                  <strong>
                    {results.ridge?.spacing_pixels !== null &&
                    results.ridge?.spacing_pixels !== undefined
                      ? `${formatNumber(
                          results.ridge.spacing_pixels,
                          2
                        )} px`
                      : "Not detected"}
                  </strong>
                  <span>
                    pixels
                  </span>
                </div>

              </div>

            </div>


            {/* Image Quality */}

            <div className="card result-card">

              <div className="card-title-row">

                <div>
                  <h3>Image Quality</h3>

                  <p>
                    Measured image characteristics within the
                    segmented foreground
                  </p>
                </div>

                <span className="status-badge detected">
                  Measured
                </span>

              </div>


              <div className="metric-grid">

                <div className="metric-box">
                  <span>Mean intensity</span>
                  <strong>
                    {formatNumber(
                      results.image_quality?.mean_intensity
                    )}
                  </strong>
                </div>

                <div className="metric-box">
                  <span>Intensity deviation</span>
                  <strong>
                    {formatNumber(
                      results.image_quality?.intensity_std
                    )}
                  </strong>
                </div>

                <div className="metric-box">
                  <span>Local variance</span>
                  <strong>
                    {formatNumber(
                      results.image_quality?.local_variance
                    )}
                  </strong>
                </div>

                <div className="metric-box">
                  <span>Foreground pixels</span>
                  <strong>
                    {displayValue(
                      results.image_quality?.foreground_pixels
                    )}
                  </strong>
                </div>

              </div>

            </div>


            {/* Minutiae */}

            <div className="card result-card">

              <div className="card-title-row">

                <div>
                  <h3>Minutiae</h3>

                  <p>
                    Ridge endings and bifurcation measurements
                  </p>
                </div>

                <span className="status-badge unavailable">
                  Not validated
                </span>

              </div>


              <div className="feature-table">

                <div className="table-row table-header">
                  <span>Feature</span>
                  <span>Value</span>
                  <span>Status</span>
                </div>

                <div className="table-row">
                  <span>Total minutiae</span>
                  <strong>
                    {displayValue(
                      results.minutiae?.total
                    )}
                  </strong>
                  <span>
                    Not detected
                  </span>
                </div>

                <div className="table-row">
                  <span>Ridge endings</span>
                  <strong>
                    {displayValue(
                      results.minutiae?.ridge_endings
                    )}
                  </strong>
                  <span>
                    Not detected
                  </span>
                </div>

                <div className="table-row">
                  <span>Bifurcations</span>
                  <strong>
                    {displayValue(
                      results.minutiae?.bifurcations
                    )}
                  </strong>
                  <span>
                    Not detected
                  </span>
                </div>

              </div>

              <div className="info-note">
                Minutiae are currently withheld because the
                experimental detector has not reached a reliable
                validation level.
              </div>

            </div>


            {/* Transparency */}

            <div className="transparency-card">

              <div className="transparency-icon">
                i
              </div>

              <div>

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
                  This application performs fingerprint image
                  feature extraction only. It does not identify
                  a person, compare fingerprints against a
                  database, or make forensic authentication
                  decisions.
                </p>

              </div>

            </div>

          </section>
        )}

      </main>

    </div>
  )
}

export default App