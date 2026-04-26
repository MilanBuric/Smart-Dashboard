const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Footer, Header
} = require('docx');
const fs = require('fs');

const BLUE   = "1E3A5F";
const LBLUE  = "2563EB";
const DARK   = "0F172A";
const GRAY   = "475569";
const LGRAY  = "F1F5F9";
const WHITE  = "FFFFFF";
const GREEN  = "065F46";
const AMBER  = "78350F";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CBD5E1" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: LBLUE, space: 6 } },
    children: [new TextRun({ text, bold: true, size: 32, color: DARK, font: "Arial" })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, size: 26, color: LBLUE, font: "Arial" })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 22, color: "1E3A5F", font: "Arial" })]
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 100 },
    children: [new TextRun({ text, size: 22, color: DARK, font: "Arial", ...opts })]
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 22, color: DARK, font: "Arial" })]
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 22, color: DARK, font: "Arial" })]
  });
}

function spacer(lines = 1) {
  return new Paragraph({ children: [new TextRun({ text: "", size: lines * 12 })] });
}

function highlightBox(title, text, color = "EFF6FF", borderColor = LBLUE) {
  return new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [9026],
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 6, color: borderColor },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: borderColor },
      left:   { style: BorderStyle.SINGLE, size: 6, color: borderColor },
      right:  { style: BorderStyle.SINGLE, size: 2, color: borderColor },
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: noBorders,
            shading: { fill: color, type: ShadingType.CLEAR },
            margins: { top: 160, bottom: 160, left: 200, right: 200 },
            width: { size: 9026, type: WidthType.DXA },
            children: [
              new Paragraph({
                spacing: { before: 0, after: 80 },
                children: [new TextRun({ text: title, bold: true, size: 22, font: "Arial", color: "1E3A5F" })]
              }),
              new Paragraph({
                spacing: { before: 0, after: 0 },
                children: [new TextRun({ text, size: 21, font: "Arial", color: DARK })]
              })
            ]
          })
        ]
      })
    ]
  });
}

function twoColTable(rows, headerRow) {
  const makeCell = (text, isHeader = false, shade = WHITE) =>
    new TableCell({
      borders,
      width: { size: 4513, type: WidthType.DXA },
      shading: { fill: shade, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text, size: 20, font: "Arial",
          bold: isHeader, color: isHeader ? WHITE : DARK })]
      })]
    });

  const tableRows = [];
  if (headerRow) {
    tableRows.push(new TableRow({
      children: headerRow.map(h => new TableCell({
        borders,
        width: { size: 4513, type: WidthType.DXA },
        shading: { fill: "1E3A5F", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({
          children: [new TextRun({ text: h, size: 20, font: "Arial", bold: true, color: WHITE })]
        })]
      }))
    }));
  }
  rows.forEach(([c1, c2], i) => {
    const shade = i % 2 === 0 ? WHITE : "F8FAFC";
    tableRows.push(new TableRow({
      children: [makeCell(c1, false, shade), makeCell(c2, false, shade)]
    }));
  });

  return new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [4513, 4513],
    rows: tableRows
  });
}

function threeColTable(rows, headerRow) {
  const col = Math.floor(9026 / 3);
  const makeCell = (text, shade = WHITE) =>
    new TableCell({
      borders,
      width: { size: col, type: WidthType.DXA },
      shading: { fill: shade, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text, size: 20, font: "Arial", color: DARK })]
      })]
    });

  const tableRows = [];
  if (headerRow) {
    tableRows.push(new TableRow({
      children: headerRow.map(h => new TableCell({
        borders,
        width: { size: col, type: WidthType.DXA },
        shading: { fill: "1E3A5F", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({
          children: [new TextRun({ text: h, size: 20, font: "Arial", bold: true, color: WHITE })]
        })]
      }))
    }));
  }
  rows.forEach(([c1,c2,c3], i) => {
    const shade = i % 2 === 0 ? WHITE : "F8FAFC";
    tableRows.push(new TableRow({ children: [makeCell(c1,shade),makeCell(c2,shade),makeCell(c3,shade)] }));
  });

  return new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [col, col, col],
    rows: tableRows
  });
}

// ==================== DOCUMENT ====================

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ]
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ]
      },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: DARK },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: LBLUE },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "1E3A5F" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E1", space: 6 } },
            children: [
              new TextRun({ text: "SmartGrid AI Dashboard  —  Hypothesis Methodology", size: 18, color: GRAY, font: "Arial" }),
              new TextRun({ text: "  |  Confidential", size: 18, color: "94A3B8", font: "Arial" }),
            ]
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E1", space: 6 } },
            children: [
              new TextRun({ text: `Generated: ${new Date().toLocaleDateString('en-GB', { year:'numeric', month:'long', day:'numeric' })}   `, size: 18, color: GRAY, font: "Arial" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GRAY, font: "Arial" }),
              new TextRun({ text: " / ", size: 18, color: GRAY, font: "Arial" }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: GRAY, font: "Arial" }),
            ]
          })
        ]
      })
    },
    children: [

      // ==================== TITLE PAGE ====================
      spacer(4),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "SmartGrid AI Dashboard", bold: true, size: 56, font: "Arial", color: DARK })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 480 },
        children: [new TextRun({ text: "Hypothesis Methodology Document", size: 32, font: "Arial", color: LBLUE })]
      }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [9026],
        rows: [new TableRow({ children: [new TableCell({
          borders: noBorders,
          shading: { fill: "EFF6FF", type: ShadingType.CLEAR },
          margins: { top: 320, bottom: 320, left: 400, right: 400 },
          width: { size: 9026, type: WidthType.DXA },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before:0, after:80 },
              children: [new TextRun({ text: "H", bold:true, size:28, font:"Arial", color: LBLUE }),
                         new TextRun({ text: "  —  AI Integration in Data Preprocessing", size:24, font:"Arial", color: DARK })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before:0, after:80 },
              children: [new TextRun({ text: "H1", bold:true, size:28, font:"Arial", color: LBLUE }),
                         new TextRun({ text: "  —  Verification of Control Panel Elements", size:24, font:"Arial", color: DARK })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before:0, after:0 },
              children: [new TextRun({ text: "H2", bold:true, size:28, font:"Arial", color: LBLUE }),
                         new TextRun({ text: "  —  Quality Assessment in Control Panel Correction", size:24, font:"Arial", color: DARK })] }),
          ]
        })})]})
      }),

      spacer(2),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: `Version 1.0  ·  ${new Date().toLocaleDateString('en-GB', { year:'numeric', month:'long' })}`, size: 20, color: GRAY, font: "Arial" })]
      }),

      // Page break
      new Paragraph({ pageBreakBefore: true, children: [new TextRun("")] }),

      // ==================== 1. INTRODUCTION ====================
      heading1("1. Introduction and Scope"),
      body("This document describes the research methodology behind three hypotheses that guide the development of the SmartGrid AI Dashboard — a real-time monitoring and analysis system for electrical grid sensor data."),
      spacer(),
      body("The hypotheses address three distinct but interconnected challenges:"),
      bullet("H: Whether artificial intelligence can be meaningfully integrated into the data preprocessing pipeline of a control panel system"),
      bullet("H1: Whether the individual elements of a control panel can be systematically verified for correctness and completeness"),
      bullet("H2: Whether a quantitative quality assessment can be performed on corrected control panel data"),
      spacer(),
      highlightBox(
        "Important Note on Technology Independence",
        "Hypotheses H1 and H2 are intentionally technology-agnostic. The methodologies described in this document apply regardless of the underlying data source (CSV files, PLC systems, MQTT brokers, REST APIs, industrial SCADA systems) or the implementation platform (Python, Java, C#, cloud services). The validation and quality scoring logic is described at a conceptual level that transcends any specific technology stack."
      ),
      spacer(),

      // ==================== 2. H ====================
      heading1("2. Hypothesis H — AI Integration in Data Preprocessing"),

      heading2("2.1 Hypothesis Statement"),
      highlightBox(
        "H",
        "It is possible to integrate artificial intelligence in the data preprocessing stage of a control panel system to improve data quality, detect anomalies, and prepare sensor data for monitoring and analysis.",
        "EFF6FF", LBLUE
      ),
      spacer(),

      heading2("2.2 Motivation"),
      body("Traditional preprocessing of sensor data relies on manually defined thresholds and statistical rules (e.g. IQR bounds, z-score limits). While effective for known failure modes, these methods:"),
      bullet("Require domain expert knowledge to configure thresholds"),
      bullet("Cannot detect complex multivariate anomalies involving interactions between multiple sensors"),
      bullet("Do not adapt as the system's operating envelope changes over time"),
      bullet("Treat each sensor independently, missing systemic patterns"),
      spacer(),
      body("Machine learning-based preprocessing addresses these limitations by learning the normal operating envelope directly from the data."),
      spacer(),

      heading2("2.3 AI Methods Implemented"),
      spacer(),

      heading3("2.3.1 Isolation Forest"),
      body("Isolation Forest is an unsupervised tree-based ML algorithm that detects anomalies by randomly partitioning the feature space. The core insight is that anomalies are isolated more quickly than normal points — they require fewer splits to be isolated in the tree."),
      spacer(),
      twoColTable([
        ["Algorithm type",   "Ensemble of random isolation trees"],
        ["Input",            "Multivariate sensor readings (any combination of features)"],
        ["Output",           "Binary label (normal / anomaly) + continuous anomaly score"],
        ["Key parameter",    "contamination — expected fraction of anomalies in dataset"],
        ["Advantage",        "Scales well to high-dimensional data; no distributional assumptions"],
        ["Limitation",       "Less effective for very high-dimensional sparse data"],
      ], ["Property", "Detail"]),
      spacer(),

      heading3("2.3.2 Local Outlier Factor (LOF)"),
      body("LOF measures the local density of a data point relative to its neighbours. Points in regions of significantly lower density than their neighbours receive high LOF scores and are flagged as outliers. This makes LOF effective for detecting local anomalies that would be missed by global methods."),
      spacer(),
      twoColTable([
        ["Algorithm type",   "Density-based neighbourhood comparison"],
        ["Input",            "Multivariate sensor readings"],
        ["Output",           "Binary label + negative outlier factor score"],
        ["Key parameter",    "n_neighbors — number of neighbours to consider"],
        ["Advantage",        "Detects local anomalies; works well in heterogeneous datasets"],
        ["Limitation",       "Computationally expensive for very large datasets"],
      ], ["Property", "Detail"]),
      spacer(),

      heading3("2.3.3 One-Class SVM"),
      body("One-Class SVM learns a decision boundary in a high-dimensional kernel space that encloses the normal data. Points falling outside this boundary are classified as anomalies. It is particularly effective when the normal class is well-defined but anomalies are diverse."),
      spacer(),
      twoColTable([
        ["Algorithm type",   "Kernel-based one-class classification"],
        ["Input",            "Standardised multivariate sensor readings"],
        ["Output",           "Binary label + decision function score"],
        ["Key parameter",    "nu — upper bound on the fraction of outliers"],
        ["Advantage",        "Flexible decision boundary via kernel trick (RBF, polynomial)"],
        ["Limitation",       "Sensitive to feature scaling; slower for large datasets"],
      ], ["Property", "Detail"]),
      spacer(),

      heading3("2.3.4 PCA Reconstruction Error"),
      body("Principal Component Analysis (PCA) projects the data into a lower-dimensional space capturing the main sources of variation. Normal data can be accurately reconstructed from these components. Anomalous readings that deviate from the principal patterns exhibit high reconstruction error and are flagged."),
      spacer(),
      twoColTable([
        ["Algorithm type",   "Linear dimensionality reduction + reconstruction"],
        ["Input",            "Standardised multivariate sensor readings"],
        ["Output",           "Per-sample reconstruction error + binary anomaly label"],
        ["Key parameter",    "n_components — number of principal components to retain"],
        ["Advantage",        "Detects multivariate anomalies; provides interpretable variance explanation"],
        ["Limitation",       "Assumes linear relationships between features"],
      ], ["Property", "Detail"]),
      spacer(),

      heading2("2.4 Preprocessing Pipeline"),
      numbered("Raw sensor data ingested from CSV (or any data source)"),
      numbered("Missing value detection and interpolation (linear, polynomial, spline)"),
      numbered("Feature normalisation (StandardScaler applied before SVM and PCA)"),
      numbered("ML model trained on full dataset in unsupervised mode (no labels required)"),
      numbered("Anomaly scores computed and thresholded to produce binary labels"),
      numbered("Results visualised alongside raw data for operator review"),
      spacer(),

      heading2("2.5 Validation of Hypothesis H"),
      body("H is considered validated if:"),
      bullet("At least one ML model successfully identifies known injected anomalies in the synthetic test dataset"),
      bullet("ML detection finds anomalies that the statistical IQR/z-score baseline does not flag (or vice versa), demonstrating complementary value"),
      bullet("The multivariate methods (PCA, Isolation Forest) identify cross-sensor patterns invisible to univariate methods"),
      bullet("Processing time is acceptable for near-real-time use (< 2 seconds per model run on 500 rows)"),
      spacer(),

      // ==================== 3. H1 ====================
      new Paragraph({ pageBreakBefore: true, children: [new TextRun("")] }),
      heading1("3. Hypothesis H1 — Verification of Control Panel Elements"),

      heading2("3.1 Hypothesis Statement"),
      highlightBox(
        "H1",
        "It is possible to systematically verify the elements of a control panel — checking that each measured parameter is present, within acceptable bounds, and logically consistent with related parameters — regardless of the technology used to implement the control panel.",
        "F0FDF4", "059669"
      ),
      spacer(),

      heading2("3.2 Technology Independence"),
      body("H1 explicitly does not depend on any specific technology. The verification methodology described below applies equally to:"),
      spacer(),
      threeColTable([
        ["Embedded / Industrial",  "Python / Web",        "Cloud / API"],
        ["PLC (Siemens, Allen-Bradley)", "Streamlit dashboard", "REST API endpoints"],
        ["SCADA systems",          "Pandas DataFrames",   "AWS IoT / Azure IoT Hub"],
        ["Arduino / ESP32 sensors","CSV file monitoring", "MQTT broker subscribers"],
        ["Modbus RTU/TCP",         "SQLite / PostgreSQL", "InfluxDB time-series DB"],
      ], ["Platform Type", "Implementation", "Data Infrastructure"]),
      spacer(),
      body("The verification rules operate on abstract data records with named fields and numeric values. The source of those records is irrelevant to the methodology."),
      spacer(),

      heading2("3.3 Verification Dimensions"),
      spacer(),

      heading3("3.3.1 Range Validation"),
      body("Each measured parameter has a defined acceptable operating range [min, max]. A reading is considered valid if and only if its value falls within this range. The validity score for a parameter is:"),
      spacer(),
      highlightBox(
        "Validity Score Formula",
        "Validity Score (%) = (1 - invalid_readings / total_readings) × 100\n\nWhere invalid_readings = count of readings outside [min, max]",
        "F8FAFC", "64748B"
      ),
      spacer(),
      body("Standard acceptable ranges for electrical grid parameters:"),
      spacer(),
      threeColTable([
        ["Voltage",             "210 V",      "250 V"],
        ["Current",             "0 A",        "25 A"],
        ["Frequency",           "49.5 Hz",    "50.5 Hz"],
        ["Active Power",        "0 W",        "5000 W"],
        ["Power Factor",        "0.85",       "1.00"],
        ["Cos Phi",             "0.85",       "1.00"],
      ], ["Parameter", "Min Acceptable", "Max Acceptable"]),
      spacer(),

      heading3("3.3.2 Logical Consistency Validation"),
      body("Beyond individual range checks, logical relationships between parameters must hold simultaneously. These rules are derived from physical laws:"),
      spacer(),
      twoColTable([
        ["Active Power ≤ Apparent Power",   "By definition: P ≤ S (power triangle)"],
        ["Power Factor = P / S",             "Must hold within rounding tolerance"],
        ["Voltage > 0 when Current > 0",     "No current without voltage (resistive load)"],
        ["Reactive Power ≥ 0",              "Non-negative reactive component (inductive load)"],
        ["Cos Phi ≈ Power Factor",          "Both measures of the same quantity — must agree"],
      ], ["Rule", "Physical Justification"]),
      spacer(),

      heading3("3.3.3 Sensor Availability (Data Completeness)"),
      body("For each sensor channel, the data availability score measures what fraction of expected readings were actually received:"),
      spacer(),
      highlightBox(
        "Availability Score Formula",
        "Availability (%) = (1 - missing_readings / total_expected_readings) × 100\n\nThreshold levels:  ≥ 95% = Healthy  |  80–95% = Warning  |  < 80% = Critical",
        "F8FAFC", "64748B"
      ),
      spacer(),

      heading2("3.4 Verification Process"),
      numbered("Define acceptable ranges per parameter (domain knowledge input)"),
      numbered("Run range validation — compute validity score per channel"),
      numbered("Run consistency checks — identify physically impossible value combinations"),
      numbered("Run sensor health check — compute availability per sensor channel"),
      numbered("Aggregate results into a verification report with pass/fail per check"),
      numbered("Flag anomalies to the operator with specific, actionable messages"),
      spacer(),

      heading2("3.5 Validation of Hypothesis H1"),
      body("H1 is considered validated if:"),
      bullet("The system correctly flags all out-of-range values in a test dataset with known violations"),
      bullet("The logical consistency checks detect physically impossible reading combinations"),
      bullet("The sensor health check accurately reports missing data rates"),
      bullet("The methodology can be described in pseudocode that maps directly to non-Python implementations (PLC ladder logic, SQL queries, etc.)"),
      spacer(),

      heading2("3.6 Platform-Agnostic Pseudocode"),
      highlightBox(
        "H1 — Generic Verification Algorithm (pseudocode)",
        "FOR each sensor_channel IN control_panel:\n    readings = fetch_readings(sensor_channel, time_window)\n    \n    // Range validation\n    invalid = [r FOR r IN readings IF r < min_range OR r > max_range]\n    validity_score = (1 - len(invalid) / len(readings)) * 100\n    \n    // Availability\n    expected = expected_reading_count(time_window, sample_rate)\n    availability = (len(readings) / expected) * 100\n    \n    report.add(sensor_channel, validity_score, availability)\n\nFOR each (param_A, param_B, rule) IN consistency_rules:\n    violations = [row FOR row IN data IF NOT rule(row[param_A], row[param_B])]\n    report.add_consistency(param_A, param_B, len(violations))",
        "F8FAFC", "64748B"
      ),
      spacer(),

      // ==================== 4. H2 ====================
      new Paragraph({ pageBreakBefore: true, children: [new TextRun("")] }),
      heading1("4. Hypothesis H2 — Quality Assessment in Control Panel Correction"),

      heading2("4.1 Hypothesis Statement"),
      highlightBox(
        "H2",
        "It is possible to assess the quality of control panel data — both before and after correction — using a quantitative, weighted multi-dimensional framework that is independent of the technology used to store or process the data.",
        "FFFBEB", "D97706"
      ),
      spacer(),

      heading2("4.2 Technology Independence"),
      body("Like H1, H2 is technology-agnostic. The quality scoring dimensions (completeness, accuracy, stability) are abstract concepts applicable to any monitoring system. The same scoring formula can be implemented in:"),
      bullet("SQL queries against a time-series database"),
      bullet("Python/pandas on CSV data"),
      bullet("JavaScript processing JSON from a REST API"),
      bullet("PLC diagnostic routines in ladder logic"),
      bullet("Excel formulas on exported spreadsheet data"),
      spacer(),

      heading2("4.3 Quality Framework"),
      body("The overall quality score is a weighted combination of three dimensions:"),
      spacer(),
      threeColTable([
        ["Completeness",      "What fraction of expected readings are present?",                  "30%"],
        ["Accuracy",          "How many readings fall outside statistically expected bounds?",     "30%"],
        ["Stability",         "Is the coefficient of variation within acceptable limits?",        "40%"],
      ], ["Dimension", "Definition", "Weight"]),
      spacer(),
      highlightBox(
        "Overall Quality Score Formula",
        "Q = 0.30 × Completeness + 0.30 × Accuracy + 0.40 × Stability\n\nAll components are normalised to [0, 100]. Final score Q ∈ [0, 100].",
        "F8FAFC", "64748B"
      ),
      spacer(),

      heading2("4.4 Dimension Definitions"),
      spacer(),

      heading3("4.4.1 Completeness (Weight: 30%)"),
      body("Measures the fraction of expected data that is actually present, across all sensor channels:"),
      highlightBox(
        "Completeness Formula",
        "Completeness (%) = (1 - total_missing / (n_rows × n_sensors)) × 100\n\nPerfect completeness = 100%. A single missing value in 500 rows × 9 sensors reduces completeness by ~0.22%.",
        "F8FAFC", "94A3B8"
      ),
      spacer(),

      heading3("4.4.2 Accuracy / Outlier Score (Weight: 30%)"),
      body("Measures what fraction of readings are statistically normal (within 3 standard deviations of the mean):"),
      highlightBox(
        "Accuracy Formula",
        "For each sensor channel:\n    z_score(x) = |x - mean| / std\n    outlier_fraction = count(z_score > 3) / n_rows\n\nAccuracy = 100 - (sum of outlier_fractions across all channels × 100)",
        "F8FAFC", "94A3B8"
      ),
      spacer(),

      heading3("4.4.3 Stability (Weight: 40%)"),
      body("The coefficient of variation (CV = std / mean) measures relative variability. Channels with CV > 0.5 are considered unstable and penalise the stability score:"),
      highlightBox(
        "Stability Formula",
        "Stability = 100\nFor each sensor channel:\n    CV = std / mean\n    IF CV > 0.5: Stability -= CV × 10\n\nStability = max(Stability, 0)  // Clamp to [0, 100]",
        "F8FAFC", "94A3B8"
      ),
      spacer(),

      heading2("4.5 Quality Rating Scale"),
      spacer(),
      twoColTable([
        ["90 – 100",  "Excellent — Data quality is optimal. No action required."],
        ["75 – 89",   "Good — Minor issues present. Routine review recommended."],
        ["60 – 74",   "Fair — Notable quality problems. Targeted investigation needed."],
        ["0 – 59",    "Poor — Significant quality issues. Immediate corrective action required."],
      ], ["Score Range", "Rating and Recommended Action"]),
      spacer(),

      heading2("4.6 Correction Impact Assessment"),
      body("When preprocessing corrections are applied (e.g. interpolation of missing values, outlier capping), the impact must be measured and documented. For each sensor channel corrected:"),
      spacer(),
      twoColTable([
        ["Rows Changed",       "Count of rows where the corrected value differs from the original"],
        ["Percent Changed",    "(rows_changed / total_rows) × 100"],
        ["Mean Absolute Error","Average absolute difference between original and corrected values"],
      ], ["Metric", "Definition"]),
      spacer(),
      body("This ensures that corrections are minimal (small MAE) and targeted (low % changed), avoiding overcorrection of data that was not actually faulty."),
      spacer(),

      heading2("4.7 Validation of Hypothesis H2"),
      body("H2 is considered validated if:"),
      bullet("The quality score correctly differentiates between a high-quality dataset and a synthetically degraded dataset (with injected missing values and outliers)"),
      bullet("The correction impact metrics correctly quantify the effect of preprocessing on the dataset"),
      bullet("The scoring formula produces consistent, reproducible results across repeated evaluations of the same dataset"),
      bullet("The methodology can be expressed as a technology-agnostic algorithm applicable to non-Python systems"),
      spacer(),

      heading2("4.8 Platform-Agnostic Pseudocode"),
      highlightBox(
        "H2 — Generic Quality Scoring Algorithm (pseudocode)",
        "FUNCTION assess_quality(data, sensor_channels):\n    \n    // Completeness\n    missing = count_nulls(data, sensor_channels)\n    total   = n_rows(data) * len(sensor_channels)\n    completeness = (1 - missing / total) * 100\n    \n    // Accuracy\n    outlier_score = 100\n    FOR each channel IN sensor_channels:\n        z = abs_z_scores(data[channel])\n        outlier_score -= count(z > 3) / n_rows(data) * 100\n    \n    // Stability\n    stability = 100\n    FOR each channel IN sensor_channels:\n        cv = std(data[channel]) / mean(data[channel])\n        IF cv > 0.5: stability -= cv * 10\n    \n    Q = 0.30*completeness + 0.30*max(outlier_score,0) + 0.40*max(stability,0)\n    RETURN Q, completeness, outlier_score, stability",
        "F8FAFC", "64748B"
      ),
      spacer(),

      // ==================== 5. SUMMARY ====================
      new Paragraph({ pageBreakBefore: true, children: [new TextRun("")] }),
      heading1("5. Summary and Relationship Between Hypotheses"),

      body("The three hypotheses form a layered quality assurance pipeline:"),
      spacer(),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [1500, 2800, 2400, 2326],
        rows: [
          new TableRow({ children: [
            ["Hypothesis","Focus","Method","Tech-agnostic?"].map(h =>
              new TableCell({
                borders, width: { size: Math.floor(9026/4), type: WidthType.DXA },
                shading: { fill: "1E3A5F", type: ShadingType.CLEAR },
                margins: { top:80, bottom:80, left:120, right:120 },
                children: [new Paragraph({ children: [new TextRun({ text: h, size:20, bold:true, color:WHITE, font:"Arial" })] })]
              }))
          ]}),
          ...[
            ["H",  "AI Preprocessing",          "Isolation Forest, LOF, One-Class SVM, PCA", "No — uses Python/sklearn"],
            ["H1", "Element Verification",       "Range, consistency, availability checks",   "Yes — methodology only"],
            ["H2", "Quality Assessment",         "Weighted multi-dimensional scoring",         "Yes — methodology only"],
          ].map(([h,f,m,t], i) => new TableRow({ children: [h,f,m,t].map(cell =>
            new TableCell({
              borders,
              width: { size: Math.floor(9026/4), type: WidthType.DXA },
              shading: { fill: i%2===0 ? WHITE : "F8FAFC", type: ShadingType.CLEAR },
              margins: { top:80, bottom:80, left:120, right:120 },
              children: [new Paragraph({ children: [new TextRun({ text: cell, size:20, color:DARK, font:"Arial" })] })]
            })
          )}))
        ]
      }),
      spacer(),
      body("Together, the hypotheses argue that an intelligent monitoring system can: (H) automatically detect data quality issues using AI, (H1) systematically verify all control panel elements, and (H2) quantify the quality of the resulting corrected dataset — all demonstrated in the SmartGrid AI Dashboard implementation."),
      spacer(),

      heading2("5.1 Dependencies"),
      bullet("H depends on the data being preprocessed (sufficient rows for ML model training)"),
      bullet("H1 operates on raw or preprocessed data and does not depend on H"),
      bullet("H2 requires both the original data and the corrected data (output of H) for impact assessment"),
      bullet("H1 and H2 can be executed independently of H if corrections come from a different source"),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/home/claude/hypothesis_methodology.docx", buffer);
  console.log("Done");
});