import { useEffect, useRef, useState } from "react"

const API_URL = "http://127.0.0.1:8000"
const MAX_FILE_SIZE_MB = 15
const ALLOWED_TYPES = ["image/jpeg", "image/png"]

function App() {
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const cameraStreamRef = useRef(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [imageDimensions, setImageDimensions] = useState(null)
  const [grayscale, setGrayscale] = useState(false)

  const [cameraOpen, setCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [cameraReady, setCameraReady] = useState(false)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
      stopCamera()
    }
  }, [previewUrl])

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop())
      cameraStreamRef.current = null
    }
    setCameraReady(false)
  }

  const resetAnalysisState = () => {
    setError(null)
    setResults(null)
  }

  const validateImageFile = (file) => {
    if (!file) return false

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("Please upload a JPG, JPEG, or PNG image.")
      return false
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`Please choose an image smaller than ${MAX_FILE_SIZE_MB} MB.`)
      return false
    }

    return true
  }

  const loadImageDimensions = (url) => {
    return new Promise((resolve, reject) => {
      const image = new Image()

      image.onload = () => {
        resolve({
          width: image.width,
          height: image.height,
        })
      }

      image.onerror = () => reject(new Error("Could not read the selected image."))
      image.src = url
    })
  }

  const setImageFile = async (file) => {
    if (!validateImageFile(file)) return

    resetAnalysisState()

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    const objectUrl = URL.createObjectURL(file)

    try {
      const dimensions = await loadImageDimensions(objectUrl)

      setSelectedFile(file)
      setPreviewUrl(objectUrl)
      setImageDimensions(dimensions)
      setGrayscale(false)
    } catch {
      URL.revokeObjectURL(objectUrl)
      setSelectedFile(null)
      setPreviewUrl(null)
      setImageDimensions(null)
      setError("The selected image could not be read.")
    }
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (file) {
      setImageFile(file)
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const file = event.dataTransfer.files?.[0]
    if (file) {
      setImageFile(file)
    }
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const openFilePicker = () => {
    fileInputRef.current?.click()
  }

  const openCameraPicker = () => {
    // On supported mobile browsers, this opens the native camera.
    cameraInputRef.current?.click()
  }

  const openLiveCamera = async () => {
    setCameraError(null)
    setCameraOpen(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
        },
        audio: false,
      })

      cameraStreamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setCameraReady(true)
      }
    } catch {
      setCameraError(
        "Camera access was unavailable. Check your browser permission and make sure the app is running on localhost or HTTPS."
      )
    }
  }

  const captureFromCamera = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas || !cameraReady) return

    const width = video.videoWidth
    const height = video.videoHeight

    if (!width || !height) {
      setCameraError("The camera image is not ready yet. Please try again.")
      return
    }

    canvas.width = width
    canvas.height = height

    const context = canvas.getContext("2d")
    context.drawImage(video, 0, 0, width, height)

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setCameraError("Could not capture the camera image.")
          return
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
        const file = new File(
          [blob],
          `fingerprint-camera-${timestamp}.jpg`,
          { type: "image/jpeg" }
        )

        closeCamera()
        await setImageFile(file)
      },
      "image/jpeg",
      0.95
    )
  }

  const closeCamera = () => {
    stopCamera()
    setCameraOpen(false)
    setCameraError(null)
  }

  const handleNativeCameraChange = (event) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (file) {
      setImageFile(file)
    }
  }

  const createGrayscaleFile = async (file) => {
    const sourceUrl = URL.createObjectURL(file)

    try {
      const image = await loadImageDimensions(sourceUrl)
      const source = new Image()

      await new Promise((resolve, reject) => {
        source.onload = resolve
        source.onerror = reject
        source.src = sourceUrl
      })

      const canvas = document.createElement("canvas")
      canvas.width = image.width
      canvas.height = image.height

      const context = canvas.getContext("2d", { willReadFrequently: true })
      context.drawImage(source, 0, 0)

      const imageData = context.getImageData(
        0,
        0,
        image.width,
        image.height
      )

      const pixels = imageData.data

      for (let index = 0; index < pixels.length; index += 4) {
        const gray = Math.round(
          0.299 * pixels[index] +
            0.587 * pixels[index + 1] +
            0.114 * pixels[index + 2]
        )

        pixels[index] = gray
        pixels[index + 1] = gray
        pixels[index + 2] = gray
      }

      context.putImageData(imageData, 0, 0)

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/png")
      })

      if (!blob) {
        throw new Error("Could not create grayscale image.")
      }

      return new File(
        [blob],
        file.name.replace(/\.[^/.]+$/, "") + "-grayscale.png",
        { type: "image/png" }
      )
    } finally {
      URL.revokeObjectURL(sourceUrl)
    }
  }

  const getAnalysisFile = async () => {
    if (!selectedFile) return null

    if (!grayscale) {
      return selectedFile
    }

    return createGrayscaleFile(selectedFile)
  }

  const handleGrayscaleChange = (event) => {
    setGrayscale(event.target.checked)
    setError(null)
    setResults(null)
  }

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return

    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const analysisFile = await getAnalysisFile()

      if (!analysisFile) {
        throw new Error("Please select a fingerprint image first.")
      }

      const formData = new FormData()
      formData.append("file", analysisFile)

      const response = await fetch(
        `${API_URL}/analyze-fingerprint`,
        {
          method: "POST",
          body: formData,
        }
      )

      let data

      try {
        data = await response.json()
      } catch {
        throw new Error(
          "The analysis server returned an invalid response."
        )
      }

      if (!response.ok) {
        throw new Error(
          data.detail || data.error || "Unable to analyze fingerprint image."
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
    setGrayscale(false)
    setResults(null)
    setError(null)

    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }

    if (cameraInputRef.current) {
      cameraInputRef.current.value = ""
    }
  }

  const formatNumber = (value, decimals = 3) => {
    if (value === null || value === undefined || value === "") {
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

  const formatConfidence = (value) => {
    if (value === null || value === undefined) {
      return "Not available"
    }

    return `${(Number(value) * 100).toFixed(0)}%`
  }

  const statusFor = (value) => {
    return value !== null && value !== undefined
      ? "Detected"
      : "Not detected"
  }

  const downloadBlob = (blob, filename) => {
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")

    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()

    URL.revokeObjectURL(url)
  }

  const downloadJSON = () => {
    if (!results) return

    const report = {
      filename: selectedFile?.name || null,
      image_dimensions: imageDimensions,
      preprocessing: {
        grayscale_applied: grayscale,
      },
      features: results,
    }

    const blob = new Blob(
      [JSON.stringify(report, null, 2)],
      { type: "application/json" }
    )

    downloadBlob(blob, "fingerprint-analysis-report.json")
  }

  const downloadCSV = () => {
    if (!results) return

    const rows = [
      ["Category", "Feature", "Value", "Confidence / Status"],
      [
        "Pattern",
        "Pattern Type",
        displayValue(results.pattern_type),
        results.pattern_details?.confidence != null
          ? formatConfidence(results.pattern_details.confidence)
          : statusFor(results.pattern_type),
      ],
      [
        "Pattern",
        "Subtype",
        displayValue(results.pattern_details?.subtype),
        results.pattern_details?.confidence != null
          ? formatConfidence(results.pattern_details.confidence)
          : "Not available",
      ],
      [
        "Image Quality",
        "Mean Intensity",
        formatNumber(results.image_quality?.mean_intensity),
        "Measured",
      ],
      [
        "Image Quality",
        "Intensity Standard Deviation",
        formatNumber(results.image_quality?.intensity_std),
        "Measured",
      ],
      [
        "Image Quality",
        "Local Variance",
        formatNumber(results.image_quality?.local_variance),
        "Measured",
      ],
      [
        "Image Quality",
        "Foreground Pixels",
        displayValue(results.image_quality?.foreground_pixels),
        "Measured",
      ],
      [
        "Core",
        "X",
        displayValue(results.core?.x),
        results.core?.confidence != null
          ? formatConfidence(results.core.confidence)
          : "Not detected",
      ],
      [
        "Core",
        "Y",
        displayValue(results.core?.y),
        results.core?.confidence != null
          ? formatConfidence(results.core.confidence)
          : "Not detected",
      ],
      [
        "Delta",
        "X",
        displayValue(results.delta?.x),
        results.delta?.confidence != null
          ? formatConfidence(results.delta.confidence)
          : "Not detected",
      ],
      [
        "Delta",
        "Y",
        displayValue(results.delta?.y),
        results.delta?.confidence != null
          ? formatConfidence(results.delta.confidence)
          : "Not detected",
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
        results.ridge?.orientation_degrees != null
          ? `${formatNumber(results.ridge.orientation_degrees, 2)}°`
          : "Not detected",
        "Measured",
      ],
      [
        "Ridge",
        "Orientation Coherence",
        formatNumber(results.ridge?.orientation_coherence),
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
        results.ridge?.spacing_pixels != null
          ? `${formatNumber(results.ridge.spacing_pixels, 2)} px`
          : "Not detected",
        "Measured",
      ],
      [
        "Minutiae",
        "Total",
        displayValue(results.minutiae?.total),
        results.minutiae?.reliable ? "Reliable" : "Not reliable",
      ],
      [
        "Minutiae",
        "Ridge Endings",
        displayValue(results.minutiae?.ridge_endings),
        results.minutiae?.reliable ? "Reliable" : "Not reliable",
      ],
      [
        "Minutiae",
        "Bifurcations",
        displayValue(results.minutiae?.bifurcations),
        results.minutiae?.reliable ? "Reliable" : "Not reliable",
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

    downloadBlob(blob, "fingerprint-analysis-report.csv")
  }

  const minutiaePoints = results?.minutiae?.points || []

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="brand-mark">FP</div>

          <div>
            <h1>Fingerprint Feature Analyzer</h1>
            <p>AI-Based Fingerprint Image Feature Extraction</p>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="card upload-card">
          <div className="section-heading">
            <div>
              <span className="eyebrow">IMAGE INPUT</span>
              <h2>Provide a fingerprint image</h2>
              <p>
                Upload an existing image or capture one directly from
                your device camera before analysis.
              </p>
            </div>
          </div>

          <div className="input-options">
            <button
              type="button"
              className="input-option primary-option"
              onClick={openFilePicker}
              disabled={loading}
            >
              <span className="option-icon">↑</span>
              <span>
                <strong>Upload Image</strong>
                <small>JPG, JPEG or PNG</small>
              </span>
            </button>

            <button
              type="button"
              className="input-option"
              onClick={openLiveCamera}
              disabled={loading}
            >
              <span className="option-icon">◉</span>
              <span>
                <strong>Use Camera</strong>
                <small>Capture a fingerprint image</small>
              </span>
            </button>
          </div>

          <div
            className="drop-zone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={openFilePicker}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                openFilePicker()
              }
            }}
          >
            <div className="drop-icon">⌁</div>
            <strong>Drop an image here</strong>
            <span>or click to browse your computer</span>
          </div>

          <input
            ref={fileInputRef}
            className="hidden-input"
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            onChange={handleFileChange}
          />

          <input
            ref={cameraInputRef}
            className="hidden-input"
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleNativeCameraChange}
          />

          {selectedFile && (
            <div className="file-summary">
              <div className="file-summary-main">
                <span className="file-label">Selected image</span>
                <strong>{selectedFile.name}</strong>
              </div>

              {imageDimensions && (
                <div className="file-summary-detail">
                  <span className="file-label">Dimensions</span>
                  <strong>
                    {imageDimensions.width} × {imageDimensions.height}px
                  </strong>
                </div>
              )}
            </div>
          )}

          {previewUrl && (
            <div className="preview-wrapper">
              <div className="preview-heading">
                <div>
                  <h3>Image Preview</h3>
                  <span>
                    {grayscale
                      ? "Grayscale preview and analysis"
                      : "Original image"}
                  </span>
                </div>

                <span className="preview-status">
                  {grayscale ? "GRAYSCALE" : "COLOR / ORIGINAL"}
                </span>
              </div>

              <div className="preview-container">
                <img
                  src={previewUrl}
                  alt="Selected fingerprint"
                  className={`fingerprint-preview ${
                    grayscale ? "grayscale-preview" : ""
                  }`}
                />
              </div>

              <div className="processing-options">
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={grayscale}
                    onChange={handleGrayscaleChange}
                    disabled={loading}
                  />

                  <span className="toggle-ui" />

                  <span className="toggle-copy">
                    <strong>Convert to grayscale</strong>
                    <small>
                      Analyze a browser-generated grayscale copy of the
                      selected image.
                    </small>
                  </span>
                </label>
              </div>
            </div>
          )}

          <div className="action-row">
            <button
              className="primary-button"
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <span className="button-spinner" />
                  Analyzing fingerprint...
                </>
              ) : (
                <>
                  <span>⌕</span>
                  Analyze Fingerprint
                </>
              )}
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
                <strong>Analyzing fingerprint...</strong>
                <span>
                  Running preprocessing and feature extraction
                </span>
              </div>
            </div>
          )}
        </section>

        {error && (
          <section className="error-card">
            <div className="error-icon">!</div>
            <div>
              <strong>Analysis failed</strong>
              <p>{error}</p>
            </div>
          </section>
        )}

        {results && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <span className="eyebrow">ANALYSIS COMPLETE</span>
                <h2>Extracted Features</h2>
                <p>
                  Measurements generated from the uploaded fingerprint
                  image.
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

            <div className="results-overview">
              <div className="overview-card">
                <span>Pattern</span>
                <strong>
                  {displayValue(results.pattern_type)}
                </strong>
                <small>
                  {results.pattern_details?.confidence != null
                    ? `${formatConfidence(
                        results.pattern_details.confidence
                      )} confidence`
                    : "Not detected"}
                </small>
              </div>

              <div className="overview-card">
                <span>Minutiae</span>
                <strong>
                  {displayValue(results.minutiae?.total)}
                </strong>
                <small>
                  {results.minutiae?.reliable
                    ? "Reliable detector output"
                    : "Not validated"}
                </small>
              </div>

              <div className="overview-card">
                <span>Ridge coherence</span>
                <strong>
                  {formatNumber(
                    results.ridge?.orientation_coherence
                  )}
                </strong>
                <small>Orientation coherence</small>
              </div>

              <div className="overview-card">
                <span>Ridge spacing</span>
                <strong>
                  {results.ridge?.spacing_pixels != null
                    ? `${formatNumber(
                        results.ridge.spacing_pixels,
                        2
                      )} px`
                    : "Not detected"}
                </strong>
                <small>Estimated ridge spacing</small>
              </div>
            </div>

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
                  <span>Confidence / Status</span>
                </div>

                <div className="table-row">
                  <span>Pattern type</span>
                  <strong>
                    {displayValue(results.pattern_type)}
                  </strong>
                  <span>
                    {results.pattern_details?.confidence != null
                      ? formatConfidence(
                          results.pattern_details.confidence
                        )
                      : "Not available"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Subtype</span>
                  <strong>
                    {displayValue(results.pattern_details?.subtype)}
                  </strong>
                  <span>
                    {results.pattern_details?.method ||
                      "Not available"}
                  </span>
                </div>
              </div>
            </div>

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
                  <strong>{displayValue(results.core?.x)}</strong>
                  <span>{statusFor(results.core?.x)}</span>
                </div>

                <div className="table-row">
                  <span>Core Y</span>
                  <strong>{displayValue(results.core?.y)}</strong>
                  <span>{statusFor(results.core?.y)}</span>
                </div>

                <div className="table-row">
                  <span>Core confidence</span>
                  <strong>
                    {formatConfidence(results.core?.confidence)}
                  </strong>
                  <span>
                    {results.core?.confidence != null
                      ? "Available"
                      : "Not available"}
                  </span>
                </div>

                <div className="table-row">
                  <span>Delta X</span>
                  <strong>{displayValue(results.delta?.x)}</strong>
                  <span>{statusFor(results.delta?.x)}</span>
                </div>

                <div className="table-row">
                  <span>Delta Y</span>
                  <strong>{displayValue(results.delta?.y)}</strong>
                  <span>{statusFor(results.delta?.y)}</span>
                </div>

                <div className="table-row">
                  <span>Delta confidence</span>
                  <strong>
                    {formatConfidence(results.delta?.confidence)}
                  </strong>
                  <span>
                    {results.delta?.confidence != null
                      ? "Available"
                      : "Not available"}
                  </span>
                </div>
              </div>
            </div>

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
                  <span>peaks / pixel</span>
                </div>

                <div className="table-row">
                  <span>Dominant ridge orientation</span>
                  <strong>
                    {results.ridge?.orientation_degrees != null
                      ? `${formatNumber(
                          results.ridge.orientation_degrees,
                          2
                        )}°`
                      : "Not detected"}
                  </strong>
                  <span>degrees</span>
                </div>

                <div className="table-row">
                  <span>Orientation coherence</span>
                  <strong>
                    {formatNumber(
                      results.ridge?.orientation_coherence
                    )}
                  </strong>
                  <span>0–1</span>
                </div>

                <div className="table-row">
                  <span>Ridge frequency</span>
                  <strong>
                    {formatNumber(results.ridge?.frequency)}
                  </strong>
                  <span>cycles / pixel</span>
                </div>

                <div className="table-row">
                  <span>Ridge spacing</span>
                  <strong>
                    {results.ridge?.spacing_pixels != null
                      ? `${formatNumber(
                          results.ridge.spacing_pixels,
                          2
                        )} px`
                      : "Not detected"}
                  </strong>
                  <span>pixels</span>
                </div>
              </div>
            </div>

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

            <div className="card result-card">
              <div className="card-title-row">
                <div>
                  <h3>Minutiae</h3>
                  <p>
                    Ridge endings and bifurcation measurements
                  </p>
                </div>

                <span
                  className={
                    results.minutiae?.reliable
                      ? "status-badge detected"
                      : "status-badge unavailable"
                  }
                >
                  {results.minutiae?.reliable
                    ? "Reliable"
                    : "Not validated"}
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
                    {displayValue(results.minutiae?.total)}
                  </strong>
                  <span>
                    {results.minutiae?.reliable
                      ? "Detected"
                      : "Not reliable"}
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
                    {results.minutiae?.reliable
                      ? "Detected"
                      : "Not reliable"}
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
                    {results.minutiae?.reliable
                      ? "Detected"
                      : "Not reliable"}
                  </span>
                </div>

                {results.minutiae?.mean_confidence != null && (
                  <div className="table-row">
                    <span>Mean confidence</span>
                    <strong>
                      {formatConfidence(
                        results.minutiae.mean_confidence
                      )}
                    </strong>
                    <span>Measured</span>
                  </div>
                )}
              </div>

              {results.minutiae?.reasons?.length > 0 && (
                <div className="info-note">
                  {results.minutiae.reasons.join(" ")}
                </div>
              )}
            </div>

            {minutiaePoints.length > 0 && (
              <div className="card result-card">
                <div className="card-title-row">
                  <div>
                    <h3>Detected Minutiae Points</h3>
                    <p>
                      Individual points returned by the current
                      detector.
                    </p>
                  </div>

                  <span className="status-badge detected">
                    {minutiaePoints.length} points
                  </span>
                </div>

                <div className="minutiae-table-wrap">
                  <table className="minutiae-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Type</th>
                        <th>X</th>
                        <th>Y</th>
                        <th>Orientation</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>

                    <tbody>
                      {minutiaePoints.map((point, index) => (
                        <tr key={`${point.x}-${point.y}-${index}`}>
                          <td>{index + 1}</td>
                          <td>
                            <span
                              className={`minutia-type ${
                                point.type === "bifurcation"
                                  ? "bifurcation"
                                  : "ending"
                              }`}
                            >
                              {point.type === "bifurcation"
                                ? "Bifurcation"
                                : "Ridge ending"}
                            </span>
                          </td>
                          <td>{point.x ?? "—"}</td>
                          <td>{point.y ?? "—"}</td>
                          <td>
                            {point.orientation != null
                              ? `${formatNumber(
                                  point.orientation,
                                  3
                                )} rad`
                              : "Not detected"}
                          </td>
                          <td>
                            {formatConfidence(point.confidence)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <div className="transparency-card">
              <div className="transparency-icon">i</div>

              <div>
                <strong>Scientific transparency</strong>

                <p>
                  Features are generated by automated image analysis.
                  Confidence values represent model or measurement
                  estimates and should not be interpreted as certainty.
                </p>

                <p>
                  This application performs fingerprint image feature
                  extraction only. It does not identify a person,
                  compare fingerprints against a database, or make
                  forensic authentication decisions.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      {cameraOpen && (
        <div className="camera-backdrop" role="dialog" aria-modal="true">
          <div className="camera-modal">
            <div className="camera-header">
              <div>
                <span className="eyebrow">CAMERA INPUT</span>
                <h2>Capture fingerprint</h2>
                <p>
                  Position the fingerprint clearly inside the frame.
                </p>
              </div>

              <button
                className="camera-close"
                onClick={closeCamera}
                aria-label="Close camera"
              >
                ×
              </button>
            </div>

            <div className="camera-view">
              <video
                ref={videoRef}
                className="camera-video"
                playsInline
                muted
              />

              <div className="camera-guide">
                <div className="camera-guide-corner top-left" />
                <div className="camera-guide-corner top-right" />
                <div className="camera-guide-corner bottom-left" />
                <div className="camera-guide-corner bottom-right" />
                <span>Place fingerprint in this area</span>
              </div>
            </div>

            {cameraError && (
              <div className="camera-error">
                {cameraError}
              </div>
            )}

            <div className="camera-actions">
              <button
                className="secondary-button"
                onClick={closeCamera}
              >
                Cancel
              </button>

              <button
                className="primary-button"
                onClick={captureFromCamera}
                disabled={!cameraReady}
              >
                Capture Image
              </button>
            </div>

            <canvas ref={canvasRef} className="hidden-canvas" />
          </div>
        </div>
      )}
    </div>
  )
}

export default App
