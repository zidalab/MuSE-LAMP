const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const csvPath = path.join(
  projectRoot,
  'runs',
  'segment',
  'seg_runs',
  'droplet_seg_v2',
  'results.csv'
);
const outputBase = path.join(__dirname, 'yolo_training_curves_best_epoch');

const rows = fs
  .readFileSync(csvPath, 'utf8')
  .trim()
  .split(/\r?\n/);
const headers = rows[0].split(',').map((value) => value.trim());
const records = rows.slice(1).map((line) => {
  const values = line.split(',');
  return Object.fromEntries(
    headers.map((header, index) => [header, Number(values[index])])
  );
});

const finiteOrNull = (value) => (Number.isFinite(value) ? value : null);
const sumFinite = (record, columns) => {
  const values = columns.map((column) => record[column]);
  return values.every(Number.isFinite)
    ? values.reduce((total, value) => total + value, 0)
    : null;
};

const trainLossColumns = [
  'train/box_loss',
  'train/seg_loss',
  'train/cls_loss',
  'train/dfl_loss',
];
const valLossColumns = [
  'val/box_loss',
  'val/seg_loss',
  'val/cls_loss',
  'val/dfl_loss',
];

// This is a segmentation experiment, so the primary curves use mask metrics (M).
const data = records.map((record) => ({
  epoch: record.epoch,
  trainLoss: sumFinite(record, trainLossColumns),
  valLoss: sumFinite(record, valLossColumns),
  precision: finiteOrNull(record['metrics/precision(M)'] * 100),
  recall: finiteOrNull(record['metrics/recall(M)'] * 100),
  map50: finiteOrNull(record['metrics/mAP50(M)'] * 100),
  map5095: finiteOrNull(record['metrics/mAP50-95(M)'] * 100),
}));

const colors = {
  blue: '#1F77B4',
  orange: '#F28E0A',
  green: '#7FB64A',
  text: '#2B2B2B',
  axis: '#747474',
  legendBorder: '#D1D1D1',
  bestEpoch: '#555555',
  white: '#FFFFFF',
};

const svgWidth = 2200;
const svgHeight = 800;
const outer = { left: 56, right: 56, top: 38, bottom: 42 };
const colGap = 88;
const panelWidth = (svgWidth - outer.left - outer.right - colGap) / 2;
const panelHeight = svgHeight - outer.top - outer.bottom;
const inner = { left: 118, right: 38, top: 72, bottom: 96 };

const bestRecord = data.reduce((best, record) =>
  record.map5095 > best.map5095 ? record : best
);

const escapeXml = (value) =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

const scaleLinear = (domainMin, domainMax, rangeMin, rangeMax) => (value) =>
  rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);

const linePath = (points, xScale, yScale) => {
  let pathData = '';
  let penDown = false;
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      penDown = false;
      continue;
    }
    const command = penDown ? 'L' : 'M';
    pathData += `${command}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)} `;
    penDown = true;
  }
  return pathData.trim();
};

const makeTicks = (min, max, step) => {
  const ticks = [];
  for (let value = min; value <= max + step * 0.25; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
};

const plotPanel = ({
  col,
  title,
  yLabel,
  yDomain,
  yTicks,
  yTickFormat,
  series,
  legend,
  bestMarkerKey,
}) => {
  const panelX = outer.left + col * (panelWidth + colGap);
  const panelY = outer.top;
  const plotX0 = panelX + inner.left;
  const plotX1 = panelX + panelWidth - inner.right;
  const plotY0 = panelY + inner.top;
  const plotY1 = panelY + panelHeight - inner.bottom;
  const xScale = scaleLinear(0, 70, plotX0, plotX1);
  const yScale = scaleLinear(yDomain[0], yDomain[1], plotY1, plotY0);
  const xTicks = [0, 20, 40, 60, 70];

  const parts = [
    `<g aria-label="${escapeXml(title)}">`,
    `<text class="panel-title" x="${panelX + panelWidth / 2}" y="${panelY + 38}" text-anchor="middle">${escapeXml(title)}</text>`,
    `<line class="axis" x1="${plotX0}" y1="${plotY0}" x2="${plotX0}" y2="${plotY1}"/>`,
    `<line class="axis" x1="${plotX0}" y1="${plotY1}" x2="${plotX1}" y2="${plotY1}"/>`,
  ];

  for (const tick of xTicks) {
    const x = xScale(tick);
    parts.push(
      `<line class="tick" x1="${x}" y1="${plotY1}" x2="${x}" y2="${plotY1 + 11}"/>`,
      `<text class="tick-label" x="${x}" y="${plotY1 + 42}" text-anchor="middle">${tick}</text>`
    );
  }

  for (const tick of yTicks) {
    const y = yScale(tick);
    parts.push(
      `<line class="tick" x1="${plotX0 - 11}" y1="${y}" x2="${plotX0}" y2="${y}"/>`,
      `<text class="tick-label" x="${plotX0 - 20}" y="${y + 9}" text-anchor="end">${escapeXml(yTickFormat(tick))}</text>`
    );
  }

  parts.push(
    `<text class="axis-label" x="${(plotX0 + plotX1) / 2}" y="${panelY + panelHeight - 18}" text-anchor="middle">Epoch</text>`,
    `<text class="axis-label" x="${panelX + 29}" y="${(plotY0 + plotY1) / 2}" text-anchor="middle" transform="rotate(-90 ${panelX + 29} ${(plotY0 + plotY1) / 2})">${escapeXml(yLabel)}</text>`
  );

  const bestX = xScale(bestRecord.epoch);
  parts.push(
    `<line class="best-epoch-line" x1="${bestX}" y1="${plotY0}" x2="${bestX}" y2="${plotY1}"/>`,
    `<text class="best-epoch-label" x="${bestX - 13}" y="${plotY0 + 31}" text-anchor="end">Best epoch = ${bestRecord.epoch}</text>`
  );

  for (const item of series) {
    const points = data.map((record) => ({ x: record.epoch, y: record[item.key] }));
    parts.push(
      `<path class="data-line" d="${linePath(points, xScale, yScale)}" stroke="${item.color}"/>`
    );
  }

  if (bestMarkerKey) {
    const markerSeries = series.find((item) => item.key === bestMarkerKey);
    const markerValue = bestRecord[bestMarkerKey];
    const markerY = yScale(markerValue);
    parts.push(
      `<circle class="best-marker" cx="${bestX}" cy="${markerY}" r="9" fill="${markerSeries.color}"/>`,
      `<text class="best-value-label" x="${bestX - 15}" y="${markerY - 17}" text-anchor="end">${markerValue.toFixed(2)}%</text>`
    );
  }

  if (legend && legend.length > 0) {
    const boxWidth = Math.max(...legend.map((item) => item.label.length)) * 16 + 112;
    const boxHeight = legend.length * 48 + 24;
    const boxX = plotX1 - boxWidth - 22;
    const boxY = plotY1 - boxHeight - 24;
    parts.push(
      `<rect class="legend-box" x="${boxX}" y="${boxY}" width="${boxWidth}" height="${boxHeight}"/>`
    );
    legend.forEach((item, index) => {
      const y = boxY + 31 + index * 48;
      parts.push(
        `<line class="legend-line" x1="${boxX + 25}" y1="${y}" x2="${boxX + 75}" y2="${y}" stroke="${item.color}"/>`,
        `<text class="legend-label" x="${boxX + 92}" y="${y + 9}">${escapeXml(item.label)}</text>`
      );
    });
  }

  parts.push('</g>');
  return parts.join('\n');
};

const lossValues = data
  .flatMap((record) => [record.trainLoss, record.valLoss])
  .filter(Number.isFinite);
const lossMax = Math.ceil(Math.max(...lossValues));

const panels = [
  plotPanel({
    col: 0,
    title: '(a) Total Loss',
    yLabel: 'Total loss',
    yDomain: [0, lossMax],
    yTicks: makeTicks(0, lossMax, 1),
    yTickFormat: (value) => value.toFixed(0),
    series: [
      { key: 'trainLoss', color: colors.blue },
      { key: 'valLoss', color: colors.orange },
    ],
    legend: [
      { label: 'Train', color: colors.blue },
      { label: 'Validation', color: colors.orange },
    ],
  }),
  plotPanel({
    col: 1,
    title: '(d) mAP',
    yLabel: 'mAP (%)',
    yDomain: [0, 100],
    yTicks: makeTicks(0, 100, 20),
    yTickFormat: (value) => value.toFixed(0),
    series: [
      { key: 'map50', color: colors.blue },
      { key: 'map5095', color: colors.orange },
    ],
    legend: [
      { label: 'mAP@0.5', color: colors.blue },
      { label: 'mAP@0.5:0.95', color: colors.orange },
    ],
    bestMarkerKey: 'map5095',
  }),
];

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" role="img" aria-labelledby="title description">
  <title id="title">YOLO segmentation training curves</title>
  <desc id="description">Two panels show total training and validation loss and mask mAP over 70 epochs, with the best epoch marked.</desc>
  <rect width="100%" height="100%" fill="${colors.white}"/>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: ${colors.text}; }
    .panel-title { font-size: 34px; font-weight: 500; }
    .axis-label { font-size: 31px; font-weight: 400; }
    .tick-label { font-size: 26px; font-weight: 400; }
    .axis, .tick { stroke: ${colors.axis}; stroke-width: 3; shape-rendering: geometricPrecision; }
    .data-line { fill: none; stroke-width: 5; stroke-linejoin: round; stroke-linecap: round; }
    .legend-box { fill: ${colors.white}; fill-opacity: 0.94; stroke: ${colors.legendBorder}; stroke-width: 2.5; }
    .legend-line { stroke-width: 7; stroke-linecap: square; }
    .legend-label { font-size: 27px; font-weight: 400; }
    .best-epoch-line { stroke: ${colors.bestEpoch}; stroke-width: 3; stroke-dasharray: 13 10; }
    .best-epoch-label, .best-value-label {
      font-size: 25px;
      font-weight: 500;
      fill: ${colors.bestEpoch};
      stroke: ${colors.white};
      stroke-width: 10;
      stroke-linejoin: round;
      paint-order: stroke fill;
    }
    .best-marker { stroke: ${colors.white}; stroke-width: 4; }
  </style>
  ${panels.join('\n')}
</svg>
`;

fs.writeFileSync(`${outputBase}.svg`, svg, 'utf8');

async function exportRaster() {
  let sharp;
  try {
    sharp = require('sharp');
  } catch (error) {
    const bundledSharp = path.join(
      process.env.USERPROFILE || '',
      '.cache',
      'codex-runtimes',
      'codex-primary-runtime',
      'dependencies',
      'node',
      'node_modules',
      'sharp'
    );
    try {
      sharp = require(bundledSharp);
    } catch (bundledError) {
      console.warn('sharp is not available; SVG was generated, raster exports were skipped.');
      return;
    }
  }

  await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .withMetadata({ density: 600 })
    .toFile(`${outputBase}.png`);

  await sharp(Buffer.from(svg))
    .tiff({ compression: 'lzw', quality: 100 })
    .withMetadata({ density: 600 })
    .toFile(`${outputBase}.tiff`);
}

exportRaster()
  .then(() => console.log(`Generated ${outputBase}.{svg,png,tiff}`))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
