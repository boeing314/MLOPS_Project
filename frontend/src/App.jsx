import { useState } from "react";

const FIELDS = [
  { key: "age",      label: "Age",                type: "number", min: 1,   max: 120, step: 1,   placeholder: "e.g. 52",  hint: "Years" },
  { key: "sex",      label: "Sex",                type: "select", options: [{ value: 1, label: "Male" }, { value: 0, label: "Female" }] },
  { key: "cp",       label: "Chest Pain Type",    type: "select", options: [{ value: 0, label: "Typical Angina" }, { value: 1, label: "Atypical Angina" }, { value: 2, label: "Non-Anginal Pain" }, { value: 3, label: "Asymptomatic" }] },
  { key: "trestbps", label: "Resting Blood Pressure", type: "number", min: 50, max: 250, step: 1, placeholder: "e.g. 130", hint: "mm Hg" },
  { key: "chol",     label: "Serum Cholesterol",  type: "number", min: 50,  max: 700, step: 1,   placeholder: "e.g. 240", hint: "mg/dl" },
  { key: "fbs",      label: "Fasting Blood Sugar > 120 mg/dl", type: "select", options: [{ value: 1, label: "True" }, { value: 0, label: "False" }] },
  { key: "restecg",  label: "Resting ECG",        type: "select", options: [{ value: 0, label: "Normal" }, { value: 1, label: "ST-T Wave Abnormality" }, { value: 2, label: "Left Ventricular Hypertrophy" }] },
  { key: "thalachh", label: "Max Heart Rate",     type: "number", min: 50,  max: 250, step: 1,   placeholder: "e.g. 150", hint: "bpm" },
  { key: "exang",    label: "Exercise-Induced Angina", type: "select", options: [{ value: 1, label: "Yes" }, { value: 0, label: "No" }] },
  { key: "oldpeak",  label: "ST Depression",      type: "number", min: 0,   max: 10,  step: 0.1, placeholder: "e.g. 2.3", hint: "mm" },
  { key: "slope",    label: "ST Segment Slope",   type: "select", options: [{ value: 0, label: "Upsloping" }, { value: 1, label: "Flat" }, { value: 2, label: "Downsloping" }] },
  { key: "ca",       label: "Major Vessels",      type: "select", options: [{ value: 0, label: "0" }, { value: 1, label: "1" }, { value: 2, label: "2" }, { value: 3, label: "3" }] },
  { key: "thal",     label: "Thalassemia",        type: "select", options: [{ value: 1, label: "Normal" }, { value: 2, label: "Fixed Defect" }, { value: 3, label: "Reversible Defect" }] },
];

const API_URL = "http://localhost:8000/predict";

const initialForm = Object.fromEntries(FIELDS.map(f => [f.key, ""]));

export default function App() {
  const [form, setForm]         = useState(initialForm);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (key, value) => {
    setForm(prev => ({ ...prev, [key]: value }));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    // Validate all fields filled
    const empty = FIELDS.filter(f => form[f.key] === "");
    if (empty.length > 0) {
      setError(`Please fill in: ${empty.map(f => f.label).join(", ")}`);
      return;
    }

    setLoading(true);
    setError(null);
    setSubmitted(true);

    const payload = Object.fromEntries(
      FIELDS.map(f => [f.key, parseFloat(form[f.key])])
    );

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm(initialForm);
    setResult(null);
    setError(null);
    setSubmitted(false);
  };

  const isHighRisk = result?.prediction === 1;

  return (
    <div style={styles.root}>
      {/* Background blobs */}
      <div style={styles.blob1} />
      <div style={styles.blob2} />
      <div style={styles.blob3} />

      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.iconWrap}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="#ef4444"/>
            </svg>
          </div>
          <div>
            <h1 style={styles.title}>Heart Disease Diagnostic</h1>
            <p style={styles.subtitle}>Enter patient clinical parameters for risk assessment</p>
          </div>
        </div>

        {/* Form Grid */}
        <div style={styles.card}>
          <div style={styles.grid}>
            {FIELDS.map((field) => (
              <div key={field.key} style={styles.fieldWrap}>
                <label style={styles.label}>{field.label}</label>
                {field.type === "select" ? (
                  <select
                    style={styles.select}
                    value={form[field.key]}
                    onChange={e => handleChange(field.key, e.target.value)}
                  >
                    <option value="">Select...</option>
                    {field.options.map(o => (
                      <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                ) : (
                  <div style={styles.inputWrap}>
                    <input
                      type="number"
                      style={styles.input}
                      value={form[field.key]}
                      placeholder={field.placeholder}
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      onChange={e => handleChange(field.key, e.target.value)}
                    />
                    {field.hint && <span style={styles.hint}>{field.hint}</span>}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Error */}
          {error && (
            <div style={styles.errorBox}>
              <span style={{ marginRight: 8 }}>⚠️</span>{error}
            </div>
          )}

          {/* Buttons */}
          <div style={styles.btnRow}>
            <button style={styles.resetBtn} onClick={handleReset}>Reset</button>
            <button
              style={{ ...styles.submitBtn, opacity: loading ? 0.7 : 1 }}
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? "Analysing..." : "Predict Risk"}
            </button>
          </div>
        </div>

        {/* Result */}
        {result && (
          <div style={{
            ...styles.resultCard,
            borderColor: isHighRisk ? "#ef4444" : "#22c55e",
            background: isHighRisk
              ? "linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%)"
              : "linear-gradient(135deg, #0a1a0a 0%, #0f2d0f 100%)",
          }}>
            <div style={styles.resultTop}>
              <div style={{
                ...styles.resultBadge,
                background: isHighRisk ? "#ef4444" : "#22c55e",
              }}>
                {isHighRisk ? "⚠ HIGH RISK" : "✓ LOW RISK"}
              </div>
              <div style={styles.probWrap}>
                <span style={styles.probLabel}>Probability</span>
                <span style={{
                  ...styles.probValue,
                  color: isHighRisk ? "#ef4444" : "#22c55e"
                }}>
                  {(result.probability * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <p style={styles.resultMsg}>
              {isHighRisk
                ? "This patient shows a high likelihood of heart disease. Immediate clinical evaluation is recommended."
                : "This patient shows a low likelihood of heart disease. Routine monitoring is advised."}
            </p>
            <p style={styles.disclaimer}>
              ⚕ This is a preliminary diagnostic aid only. Always consult a qualified cardiologist.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────────
const styles = {
  root: {
    minHeight: "100vh",
    background: "#0a0f1a",
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    position: "relative",
    overflow: "hidden",
    padding: "40px 16px",
  },
  blob1: {
    position: "fixed", top: "-100px", left: "-100px",
    width: "400px", height: "400px", borderRadius: "50%",
    background: "radial-gradient(circle, rgba(239,68,68,0.15) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  blob2: {
    position: "fixed", bottom: "-100px", right: "-100px",
    width: "500px", height: "500px", borderRadius: "50%",
    background: "radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  blob3: {
    position: "fixed", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
    width: "600px", height: "300px", borderRadius: "50%",
    background: "radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  container: {
    maxWidth: "860px",
    margin: "0 auto",
    position: "relative",
    zIndex: 1,
  },
  header: {
    display: "flex", alignItems: "center", gap: "16px",
    marginBottom: "32px",
  },
  iconWrap: {
    width: "56px", height: "56px", borderRadius: "16px",
    background: "rgba(239,68,68,0.15)",
    border: "1px solid rgba(239,68,68,0.3)",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0,
  },
  title: {
    margin: 0, fontSize: "28px", fontWeight: 700,
    color: "#f1f5f9", letterSpacing: "-0.5px",
  },
  subtitle: {
    margin: "4px 0 0", fontSize: "14px", color: "#64748b",
  },
  card: {
    background: "rgba(15, 23, 42, 0.8)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "20px",
    padding: "32px",
    backdropFilter: "blur(12px)",
    marginBottom: "24px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
    gap: "20px",
    marginBottom: "28px",
  },
  fieldWrap: {
    display: "flex", flexDirection: "column", gap: "6px",
  },
  label: {
    fontSize: "12px", fontWeight: 600, color: "#94a3b8",
    textTransform: "uppercase", letterSpacing: "0.05em",
  },
  inputWrap: {
    position: "relative", display: "flex", alignItems: "center",
  },
  input: {
    width: "100%", padding: "10px 48px 10px 14px",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "10px", color: "#f1f5f9",
    fontSize: "14px", outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.2s",
  },
  hint: {
    position: "absolute", right: "12px",
    fontSize: "11px", color: "#475569", pointerEvents: "none",
  },
  select: {
    width: "100%", padding: "10px 14px",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "10px", color: "#f1f5f9",
    fontSize: "14px", outline: "none", cursor: "pointer",
    appearance: "none",
  },
  errorBox: {
    background: "rgba(239,68,68,0.1)",
    border: "1px solid rgba(239,68,68,0.3)",
    borderRadius: "10px", padding: "12px 16px",
    color: "#fca5a5", fontSize: "13px",
    marginBottom: "20px",
  },
  btnRow: {
    display: "flex", gap: "12px", justifyContent: "flex-end",
  },
  resetBtn: {
    padding: "12px 24px", borderRadius: "10px",
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.1)",
    color: "#94a3b8", fontSize: "14px", fontWeight: 600,
    cursor: "pointer",
  },
  submitBtn: {
    padding: "12px 32px", borderRadius: "10px",
    background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
    border: "none", color: "#fff",
    fontSize: "14px", fontWeight: 700,
    cursor: "pointer", letterSpacing: "0.02em",
    boxShadow: "0 4px 24px rgba(239,68,68,0.35)",
  },
  resultCard: {
    border: "1px solid",
    borderRadius: "20px", padding: "28px",
    backdropFilter: "blur(12px)",
  },
  resultTop: {
    display: "flex", alignItems: "center",
    justifyContent: "space-between", marginBottom: "16px",
    flexWrap: "wrap", gap: "12px",
  },
  resultBadge: {
    padding: "8px 20px", borderRadius: "100px",
    color: "#fff", fontSize: "13px", fontWeight: 800,
    letterSpacing: "0.08em",
  },
  probWrap: {
    display: "flex", flexDirection: "column", alignItems: "flex-end",
  },
  probLabel: {
    fontSize: "11px", color: "#64748b",
    textTransform: "uppercase", letterSpacing: "0.05em",
  },
  probValue: {
    fontSize: "32px", fontWeight: 800, lineHeight: 1,
  },
  resultMsg: {
    color: "#cbd5e1", fontSize: "14px", lineHeight: 1.6, margin: "0 0 12px",
  },
  disclaimer: {
    color: "#475569", fontSize: "12px", margin: 0,
    borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: "12px",
  },
};