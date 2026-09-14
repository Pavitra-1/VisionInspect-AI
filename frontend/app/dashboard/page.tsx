
"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (selectedFile: File | undefined) => {
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);

    const imageUrl = URL.createObjectURL(selectedFile);
    setPreview(imageUrl);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setResult(data);
    } catch (error) {
      console.error("Upload error:", error);

      alert(
        error instanceof Error
          ? error.message
          : "Unable to connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const getSeverityClass = (severity: string) => {
    switch (severity) {
      case "Critical":
        return "text-red-400 bg-red-500/10 border-red-500/20";
      case "High":
        return "text-orange-400 bg-orange-500/10 border-orange-500/20";
      case "Medium":
        return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
      default:
        return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
    }
  };

  const getDecisionClass = (decision: string) => {
    if (decision === "FAIL") {
      return "text-red-400 bg-red-500/10 border-red-500/20";
    }

    if (decision === "MANUAL REVIEW") {
      return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
    }

    return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* HEADER */}
      <header className="border-b border-slate-800 bg-slate-950/95">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-600 text-xl font-bold">
              AI
            </div>

            <div>
              <h1 className="text-lg font-bold">VisionInspect AI</h1>
              <p className="text-xs text-slate-400">
                Manufacturing Quality Inspection
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-sm text-emerald-300">
              System Online
            </span>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <section className="mx-auto max-w-7xl px-6 py-10">
        {/* HERO */}
        <div className="mb-10">
          <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-blue-400">
            AI Vision Inspection
          </p>

          <h2 className="max-w-3xl text-4xl font-bold leading-tight md:text-5xl">
            Detect manufacturing defects
            <span className="text-blue-400"> faster and smarter.</span>
          </h2>

          <p className="mt-4 max-w-2xl text-slate-400">
            Upload a manufacturing component image and analyze it using
            computer vision, image processing, defect detection and quality
            assessment.
          </p>
        </div>

        {/* DASHBOARD */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* UPLOAD PANEL */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl lg:col-span-2">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h3 className="text-xl font-semibold">
                  Image Inspection
                </h3>

                <p className="mt-1 text-sm text-slate-400">
                  Select a manufacturing image for inspection
                </p>
              </div>

              <div className="rounded-lg bg-blue-500/10 px-3 py-2 text-xs font-medium text-blue-400">
                IMAGE ANALYSIS
              </div>
            </div>

            {/* UPLOAD AREA */}
            <label
              htmlFor="file-upload"
              className="group flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-950/60 p-6 text-center transition hover:border-blue-500 hover:bg-blue-500/5"
            >
              {preview ? (
                <img
                  src={preview}
                  alt="Selected manufacturing component"
                  className="max-h-[260px] max-w-full rounded-xl object-contain shadow-lg"
                />
              ) : (
                <>
                  <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600/10 text-3xl">
                    📷
                  </div>

                  <h4 className="text-lg font-semibold">
                    Upload inspection image
                  </h4>

                  <p className="mt-2 text-sm text-slate-400">
                    Click here to select an image
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    PNG, JPG or JPEG
                  </p>
                </>
              )}

              <input
                id="file-upload"
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={(e) =>
                  handleFileChange(e.target.files?.[0])
                }
              />
            </label>

            {/* FILE INFO */}
            {file && (
              <div className="mt-5 flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950 p-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600/10">
                    🖼️
                  </div>

                  <div>
                    <p className="max-w-[300px] truncate text-sm font-medium">
                      {file.name}
                    </p>

                    <p className="text-xs text-slate-500">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>

                <span className="rounded-full bg-emerald-500/10 px-3 py-1 text-xs text-emerald-400">
                  Ready
                </span>
              </div>
            )}

            {/* BUTTON */}
            <button
              onClick={handleUpload}
              disabled={loading || !file}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-4 font-semibold transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            >
              {loading ? (
                <>
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Analyzing Image...
                </>
              ) : (
                <>
                  🔍 Analyze for Defects
                </>
              )}
            </button>
          </div>

          {/* SYSTEM STATUS */}
          <div className="space-y-6">
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="font-semibold">Inspection System</h3>

              <div className="mt-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">
                    Backend API
                  </span>

                  <span className="text-sm font-medium text-emerald-400">
                    ● Online
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">
                    Image Processing
                  </span>

                  <span className="text-sm font-medium text-emerald-400">
                    ● Ready
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">
                    Inspection Size
                  </span>

                  <span className="text-sm font-medium">
                    128 × 128 (AI model)
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">
                    Processing
                  </span>

                  <span className="text-sm font-medium">
                    Autoencoder + ResNet18
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-blue-600/20 to-slate-900 p-6">
              <div className="mb-4 text-2xl">🏭</div>

              <h3 className="font-semibold">Quality Control</h3>

              <p className="mt-2 text-sm leading-6 text-slate-400">
                Automated visual inspection helps identify defective
                manufacturing components before they reach production.
              </p>
            </div>
          </div>
        </div>

        {/* RESULTS */}
        {result && (
          <section className="mt-10">
            <div className="mb-6">
              <p className="text-sm font-semibold uppercase tracking-widest text-blue-400">
                Analysis Complete
              </p>

              <h2 className="mt-1 text-2xl font-bold">
                Inspection Result
              </h2>
            </div>

            {/* TOP RESULT CARDS */}
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <p className="text-sm text-slate-400">
                  Defect Detection
                </p>

                <p
                  className={`mt-3 text-2xl font-bold ${
                    result.defect_detection?.detected
                      ? "text-red-400"
                      : "text-emerald-400"
                  }`}
                >
                  {result.defect_detection?.detected
                    ? "DEFECT"
                    : "NO DEFECT"}
                </p>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <p className="text-sm text-slate-400">
                  Defect Type
                </p>

                <p className="mt-3 text-xl font-bold">
                  {result.defect_detection?.defect_type || "Not detected"}
                </p>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <p className="text-sm text-slate-400">
                  Confidence
                </p>

                <p className="mt-3 text-2xl font-bold text-blue-400">
                  {result.defect_detection?.classifier_confidence !== undefined
                    ? `${result.defect_detection.classifier_confidence}%`
                    : "N/A"}
                </p>
              </div>

              <div
                className={`rounded-2xl border p-6 ${
                  result.quality_assessment?.severity_level
                    ? getSeverityClass(result.quality_assessment.severity_level)
                    : "border-slate-800 bg-slate-900"
                }`}
              >
                <p className="text-sm opacity-80">
                  Severity
                </p>

                <p className="mt-3 text-2xl font-bold">
                  {result.quality_assessment?.severity_level || "N/A"}
                </p>
              </div>
            </div>

            {/* QUALITY DECISION */}
            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <div
                className={`rounded-2xl border p-6 ${
                  result.quality_assessment?.quality_decision
                    ? getDecisionClass(result.quality_assessment.quality_decision)
                    : "border-slate-800 bg-slate-900"
                }`}
              >
                <p className="text-sm opacity-80">
                  Quality Control Decision
                </p>

                <p className="mt-3 text-3xl font-bold">
                  {result.quality_assessment?.quality_decision || "N/A"}
                </p>

                <p className="mt-2 text-sm opacity-80">
                  {result.quality_assessment?.recommended_action ||
                    "No recommendation available"}
                </p>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <p className="text-sm text-slate-400">
                  Processing Time
                </p>

                <p className="mt-3 text-3xl font-bold">
                  {result.system_performance?.processing_time_seconds !== undefined
                    ? `${result.system_performance.processing_time_seconds} seconds`
                    : "N/A"}
                </p>

                <p className="mt-2 text-sm text-slate-500">
                  Image inspection processing time
                </p>
              </div>
            </div>

            {/* IMAGE QUALITY */}
            {result.image_quality && (
              <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <h3 className="mb-5 text-lg font-semibold">
                  Image Quality Analysis
                </h3>

                <div className="grid gap-4 md:grid-cols-4">
                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      Brightness
                    </p>
                    <p className="mt-2 text-xl font-bold">
                      {result.image_quality.brightness}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      Contrast
                    </p>
                    <p className="mt-2 text-xl font-bold">
                      {result.image_quality.contrast}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      Sharpness
                    </p>
                    <p className="mt-2 text-xl font-bold">
                      {result.image_quality.sharpness}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      Quality Status
                    </p>
                    <p className="mt-2 text-xl font-bold text-blue-400">
                      {result.image_quality.quality_status}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* PREPROCESSING */}
            {result.preprocessing && (
              <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <h3 className="mb-5 text-lg font-semibold">
                  Image Processing Pipeline
                </h3>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                  {Object.entries(result.preprocessing).map(
                    ([key, value]) => (
                      <div
                        key={key}
                        className="rounded-xl bg-slate-950 p-4"
                      >
                        <p className="text-xs uppercase tracking-wider text-slate-500">
                          {key.replace(/_/g, " ")}
                        </p>

                        <p className="mt-2 text-sm font-semibold text-slate-200">
                          {String(value)}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {/* DEFECT ANALYSIS */}
            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="mb-5 text-lg font-semibold">
                Defect Analysis
              </h3>

              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Defect Location
                  </p>
                  <p className="mt-2 text-sm font-semibold">
                    {result.defect_detection?.defect_location || "N/A"}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Defect Area
                  </p>
                  <p className="mt-2 text-sm font-semibold">
                    {result.defect_detection?.defect_area_percent !== undefined
                      ? `${result.defect_detection.defect_area_percent}%`
                      : "N/A"}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Size Score
                  </p>
                  <p className="mt-2 text-sm font-semibold">
                    {result.scoring?.defect_size_score ?? "N/A"} / 100
                  </p>
                </div>

                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Bounding Box
                  </p>
                  <p className="mt-2 text-sm font-semibold">
                    {result.defect_detection?.bounding_box
                      ? JSON.stringify(result.defect_detection.bounding_box)
                      : "N/A"}
                  </p>
                </div>
              </div>
            </div>

            {/* SEVERITY SCORING */}
            {result.scoring?.overall_severity_score !== undefined && (
              <div className="mt-6 rounded-2xl border border-blue-500/20 bg-blue-500/5 p-6">
                <div className="mb-6">
                  <p className="text-sm font-semibold uppercase tracking-widest text-blue-400">
                    PDF Severity Scoring Framework
                  </p>

                  <h3 className="mt-1 text-xl font-bold">
                    Weighted Severity Assessment
                  </h3>
                </div>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  <div className="rounded-xl bg-slate-950 p-5">
                    <p className="text-sm text-slate-400">
                      Defect Size
                    </p>
                    <p className="mt-2 text-2xl font-bold">
                      {result.scoring?.defect_size_score ?? "N/A"}
                    </p>
                    <p className="mt-1 text-xs text-blue-400">
                      Weight: 30%
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-5">
                    <p className="text-sm text-slate-400">
                      Defect Location
                    </p>
                    <p className="mt-2 text-2xl font-bold">
                      {result.scoring?.defect_location_score ?? "N/A"}
                    </p>
                    <p className="mt-1 text-xs text-blue-400">
                      Weight: 25%
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-5">
                    <p className="text-sm text-slate-400">
                      Defect Type
                    </p>
                    <p className="mt-2 text-2xl font-bold">
                      {result.scoring?.defect_type_score ?? "N/A"}
                    </p>
                    <p className="mt-1 text-xs text-blue-400">
                      Weight: 25%
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-5">
                    <p className="text-sm text-slate-400">
                      Detection Confidence
                    </p>
                    <p className="mt-2 text-2xl font-bold">
                      {result.scoring?.detection_confidence_score ?? "N/A"}
                    </p>
                    <p className="mt-1 text-xs text-blue-400">
                      Weight: 20%
                    </p>
                  </div>
                </div>

                <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950 p-6 text-center">
                  <p className="text-sm text-slate-400">
                    Final Severity Score
                  </p>

                  <p className="mt-2 text-5xl font-bold text-blue-400">
                    {result.scoring.overall_severity_score}
                    <span className="text-2xl"> / 100</span>
                  </p>

                  <p className="mt-3 text-sm text-slate-400">
                    (Size × 30%) + (Location × 25%) +
                    (Type × 25%) + (Confidence × 20%)
                  </p>
                </div>
              </div>
            )}

            {/* MODEL PERFORMANCE */}
            {result.model_performance && (
              <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <h3 className="mb-5 text-lg font-semibold">
                  AI Model Performance
                </h3>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                  {[
                    ["Accuracy", result.model_performance.accuracy],
                    ["Precision", result.model_performance.precision],
                    ["Recall", result.model_performance.recall],
                    ["F1 Score", result.model_performance.f1_score],
                    ["mAP", result.model_performance.map_score],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-xl bg-slate-950 p-4"
                    >
                      <p className="text-xs uppercase text-slate-500">
                        {label}
                      </p>

                      <p className="mt-2 text-xl font-bold">
                        {value !== null && value !== undefined
                          ? value
                          : "Pending"}
                      </p>
                    </div>
                  ))}
                </div>

                {result.model_performance.evaluation_status && (
                  <p className="mt-4 text-sm text-slate-400">
                    {result.model_performance.evaluation_status}
                  </p>
                )}
              </div>
            )}

            {/* MANUFACTURING METRICS */}
            {result.manufacturing_metrics && (
              <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <h3 className="mb-5 text-lg font-semibold">
                  Manufacturing Analytics
                </h3>

                <div className="grid gap-4 md:grid-cols-3">
                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      Inspection Automation Rate
                    </p>
                    <p className="mt-2 text-xl font-bold">
                      {result.manufacturing_metrics
                        .inspection_automation_rate ?? "Pending"}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      Defect Identification Accuracy
                    </p>
                    <p className="mt-2 text-xl font-bold">
                      {result.manufacturing_metrics
                        .defect_identification_accuracy ?? "Pending"}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <p className="text-xs uppercase text-slate-500">
                      False Defect Rate
                    </p>
                    <p className="mt-2 text-xl font-bold">
                      {result.manufacturing_metrics
                        .false_defect_detection_rate ?? "Pending"}
                    </p>
                  </div>
                </div>

                {result.manufacturing_metrics.evaluation_status && (
                  <p className="mt-4 text-sm text-slate-400">
                    {result.manufacturing_metrics.evaluation_status}
                  </p>
                )}
              </div>
            )}

            {/* FILE DETAILS */}
            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="mb-5 text-lg font-semibold">
                Inspection Details
              </h3>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Message
                  </p>
                  <p className="mt-2 text-sm text-slate-200">
                    {result.message}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Filename
                  </p>
                  <p className="mt-2 break-all text-sm text-slate-200">
                    {result.filename}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Original Path
                  </p>
                  <p className="mt-2 break-all text-sm text-slate-400">
                    {result.original_path}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-950 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Processed Path
                  </p>
                  <p className="mt-2 break-all text-sm text-slate-400">
                    {result.processed_path}
                  </p>
                </div>
              </div>
            </div>
          </section>
        )}
      </section>

      {/* FOOTER */}
      <footer className="border-t border-slate-800 py-6">
        <p className="text-center text-xs text-slate-500">
          VisionInspect AI — Manufacturing Defect Detection & Quality
          Inspection System
        </p>
      </footer>
    </main>
  );
}

