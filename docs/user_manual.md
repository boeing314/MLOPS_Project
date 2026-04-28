# User Manual — Heart Disease Diagnostic System

## What is this application?

The Heart Disease Diagnostic System is a web-based tool that helps healthcare professionals
assess whether a patient may be at risk of heart disease. You enter the patient's clinical
measurements and the system gives you an instant risk prediction.

> ⚕ **Important:** This tool is a preliminary diagnostic aid only. It does not replace
> the judgment of a qualified cardiologist. Always consult a medical professional before
> making any clinical decisions.

---

## How to Access the Application

1. Open your web browser (Chrome, Firefox, or Edge recommended)
2. Go to: **http://localhost:3000**
3. The Heart Disease Diagnostic form will appear

---

## How to Use the Form

### Step 1 — Fill in the patient details

You will see 13 input fields. Fill in each one using the patient's clinical data:

| Field | What to enter |
|---|---|
| **Age** | Patient's age in years (e.g. 52) |
| **Sex** | Select Male or Female |
| **Chest Pain Type** | Select the type from the dropdown |
| **Resting Blood Pressure** | Enter in mm Hg (e.g. 130) |
| **Serum Cholesterol** | Enter in mg/dl (e.g. 240) |
| **Fasting Blood Sugar > 120 mg/dl** | Select True or False |
| **Resting ECG** | Select the result from the dropdown |
| **Max Heart Rate** | Enter in bpm (e.g. 150) |
| **Exercise-Induced Angina** | Select Yes or No |
| **ST Depression** | Enter the value in mm (e.g. 2.3) |
| **ST Segment Slope** | Select Upsloping, Flat, or Downsloping |
| **Major Vessels** | Select number of vessels (0, 1, 2, or 3) |
| **Thalassemia** | Select Normal, Fixed Defect, or Reversible Defect |

---

### Step 2 — Click "Predict Risk"

Once all fields are filled, click the **Predict Risk** button at the bottom right.

---

### Step 3 — Read the result

The result card will appear below the form showing one of two outcomes:

**High Risk (shown in red):**
```
⚠ HIGH RISK
Probability: 87.3%
This patient shows a high likelihood of heart disease.
Immediate clinical evaluation is recommended.
```

**Low Risk (shown in green):**
```
✓ LOW RISK
Probability: 12.4%
This patient shows a low likelihood of heart disease.
Routine monitoring is advised.
```

The **probability** percentage indicates how confident the model is in its prediction.

---

### Step 4 — Reset for next patient

Click the **Reset** button to clear all fields and assess a new patient.

---

## Chest Pain Type — Reference Guide

| Option | Meaning |
|---|---|
| Typical Angina | Chest pain related to heart, triggered by exertion |
| Atypical Angina | Chest pain not typical of cardiac origin |
| Non-Anginal Pain | Pain unrelated to angina |
| Asymptomatic | No chest pain |

---

## Resting ECG — Reference Guide

| Option | Meaning |
|---|---|
| Normal | No abnormalities detected |
| ST-T Wave Abnormality | Changes in ST segment or T wave |
| Left Ventricular Hypertrophy | Enlarged left ventricle |

---

## Thalassemia — Reference Guide

| Option | Meaning |
|---|---|
| Normal | No thalassemia |
| Fixed Defect | Defect present but not reversible |
| Reversible Defect | Defect that can be reversed with treatment |

---

## Troubleshooting

**"Please fill in all fields" message:**
Make sure every field has a value selected or entered before clicking Predict Risk.

**"Failed to fetch" error:**
The backend server may not be running. Contact your system administrator.

**Page not loading:**
Make sure you are connected to the hospital network and try refreshing the page.

---

## Support

For technical issues, contact your system administrator.
For clinical questions, consult a qualified cardiologist.