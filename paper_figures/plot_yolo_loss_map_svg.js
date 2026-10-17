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
const outputBase = path.join(__dirname, 'yolo_box_mask_loss_map');

const csvRows = fs.readFileSync(csvPath, 'utf8').trim().split(/\r?\n/);
const headers = csvRows[0].split(',').map((value) => value.trim());
const records = csvRows.slice(1).map((line) => {
  const values = line.split(',');
  return Object.fromEntries(
    headers.map((header, index) => [header, Number(values[index])])
  );
});

const finiteOrNull = (value) => (Number.isFinite(value) ? value : null);
const data = records.map((record) => ({
  epoch: record.epoch,
  trainBox: finiteOrNull(record['train/box_loss']),
  valBox: finiteOrNull(record['val/box_loss']),
  trainMask: finiteOrNull(record['train/seg_loss']),
  valMask: finiteOrNull(record['val/seg_loss']),
  boxMap50: finiteOrNull(record['metrics/mAP50(B)'] * 100),
  boxMap5095: finiteOrNull(record['metrics/mAP50-95(B)'] * 100),
  maskMap50: finiteOrNull(record['metrics/mAP50(M)'] * 100),
  maskMap5095: finiteOrNull(record['metrics/mAP50-95(M)'] * 100),
}));

// Best epoch is defined by the maximum mask mAP@0.5:0.95.
const bestRecord = data.reduce((best, record) =>
  record.maskMap5095 > best.maskMap5095 ? record : best
);

const palette = {
  boxTrain: '#2166AC',
  boxValidation: '#4393C3',
  maskTrain: '#B2182B',
  maskValidation: '#D6604D',
  text: '#2B2B2B',
  axis: '#747474',
  best: '#555555',
  border: '#D1D1D1',
  white: '#FFFFFF',
};

// Each 600 px panel is exactly 60% of the preceding 1000 px panel width.
const svgWidth = 1350;
const svgHeight = 780;
const outer = { left: 40, right: 40, top: 34, bottom: 34 };
const gap = 70;
const panelWidth = 600;
const panelHeight = svgHeight - outer.top - outer.bottom;
const inner = { left: 80, right: 25, top: 64, bottom: 82 };

const escapeXml = (value) =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

const scaleLinear = (domainMin, domainMax, rangeMin, rangeMax) => (value) =>
  rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);

const linePath = (points, xScale, yScale) => {
  let d = '';
  let penDown = false;
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      penDown = false;
      continue;
    }
    d += `${penDown ? 'L' : 'M'}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)} `;
    penDown = true;
  }
  return d.trim();
};

const ticks = (min, max, step) => {
  const values = [];
  for (let value = min; value <= max + step * 0.1; value += step) {
    values.push(Number(value.toFixed(10)));
  }
  return values;
};

const renderLegend = ({ items, x, y, columns = 1, columnWidth = 180 }) => {
  const rows = Math.ceil(items.length / columns);
  const width = columns * columnWidth + 24;
  const height = rows * 33 + 19;
  const parts = [
    `<g id="legend">`,
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" class="legend-box"/>`,
  ];

  items.forEach((item, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const itemX = x + 16 + column * columnWidth;
    const itemY = y + 24 + row * 33;
    parts.push(
      `<line id="legend-${item.id}" x1="${itemX}" y1="${itemY}" x2="${itemX + 34}" y2="${itemY}" stroke="${item.color}" stroke-width="4"${item.dash ? ` stroke-dasharray="${item.dash}"` : ''}/>` ,
      `<text x="${itemX + 45}" y="${itemY + 6}" class="legend-label">${escapeXml(item.label)}</text>`
    );
  });
  parts.push('</g>');
  return parts.join('\n');
};

const renderPanel = ({
  id,
  col,
  title,
  yLabel,
  yDomain,
  yTicks,
  yFormat,
  series,
  legend,
  legendPosition,
  markerKeys,
  bestValueLabel,
}) => {
  const panelX = outer.left + col * (panelWidth + gap);
  const panelY = outer.top;
  const x0 = panelX + inner.left;
  const x1 = panelX + panelWidth - inner.right;
  const y0 = panelY + inner.top;
  const y1 = panelY + panelHeight - inner.bottom;
  const xScale = scaleLinear(0, 70, x0, x1);
  const yScale = scaleLinear(yDomain[0], yDomain[1], y1, y0);
  const xTicks = [0, 20, 40, 60, 70];
  const bestX = xScale(bestRecord.epoch);
  const clipId = `${id}-plot-clip`;

  const parts = [
    `<g id="${id}">`,
    `<title>${escapeXml(title)}</title>`,
    `<defs><clipPath id="${clipId}"><rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}"/></clipPath></defs>`,
    `<text id="${id}-title" x="${panelX + panelWidth / 2}" y="${panelY + 32}" class="panel-title" text-anchor="middle">${escapeXml(title)}</text>`,
    `<g id="${id}-axes">`,
    `<line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y1}" class="axis"/>`,
    `<line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" class="axis"/>`,
  ];

  for (const tick of xTicks) {
    const x = xScale(tick);
    parts.push(
      `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y1 + 8}" class="axis"/>`,
      `<text x="${x}" y="${y1 + 31}" class="tick-label" text-anchor="middle">${tick}</text>`
    );
  }

  for (const tick of yTicks) {
    const y = yScale(tick);
    parts.push(
      `<line x1="${x0 - 8}" y1="${y}" x2="${x0}" y2="${y}" class="axis"/>`,
      `<text x="${x0 - 14}" y="${y + 6}" class="tick-label" text-anchor="end">${escapeXml(yFormat(tick))}</text>`
    );
  }

  parts.push(
    `<text id="${id}-x-label" x="${(x0 + x1) / 2}" y="${panelY + panelHeight - 14}" class="axis-label" data-axis="x" text-anchor="middle">Epoch</text>`,
    `<text id="${id}-y-label" x="${panelX + 24}" y="${(y0 + y1) / 2}" class="axis-label" data-axis="y" text-anchor="middle" transform="rotate(-90 ${panelX + 24} ${(y0 + y1) / 2})">${escapeXml(yLabel)}</text>`,
    `</g>`,
    `<g id="${id}-curves" clip-path="url(#${clipId})">`
  );

  for (const item of series) {
    const points = data.map((record) => ({ x: record.epoch, y: record[item.key] }));
    parts.push(
      `<path id="${item.id}" d="${linePath(points, xScale, yScale)}" fill="none" stroke="${item.color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"${item.dash ? ` stroke-dasharray="${item.dash}"` : ''}/>`
    );
  }

  parts.push(
    `<line id="${id}-best-epoch-line" x1="${bestX}" y1="${y0}" x2="${bestX}" y2="${y1}" class="best-line"/>`,
    `</g>`,
    `<g id="${id}-best-epoch-annotation">`,
    `<text x="${bestX - 10}" y="${y0 + 24}" class="best-label" text-anchor="end">Best epoch = ${bestRecord.epoch}</text>`
  );

  for (const key of markerKeys) {
    const item = series.find((candidate) => candidate.key === key);
    const value = bestRecord[key];
    parts.push(
      `<circle id="${item.id}-best-point" cx="${bestX}" cy="${yScale(value)}" r="5.5" fill="${item.color}" stroke="${palette.white}" stroke-width="2"/>`
    );
  }

  if (bestValueLabel) {
    const value = bestRecord[bestValueLabel.key];
    parts.push(
      `<text x="${bestX - 10}" y="${yScale(value) - 12}" class="best-label" text-anchor="end">${escapeXml(bestValueLabel.format(value))}</text>`
    );
  }

  parts.push('</g>');

  const legendX = x0 + legendPosition.x;
  const legendY = y0 + legendPosition.y;
  parts.push(renderLegend({ ...legend, x: legendX, y: legendY }), '</g>');
  return parts.join('\n');
};

const lossSeries = [
  { id: 'train-box-loss', key: 'trainBox', label: 'Train Box', color: palette.boxTrain },
  { id: 'validation-box-loss', key: 'valBox', label: 'Validation Box', color: palette.boxValidation },
  { id: 'train-mask-loss', key: 'trainMask', label: 'Train Mask', color: palette.maskTrain },
  { id: 'validation-mask-loss', key: 'valMask', label: 'Validation Mask', color: palette.maskValidation },
];

const mapSeries = [
  { id: 'box-map-50', key: 'boxMap50', label: 'Box mAP@0.5', color: palette.boxTrain },
  { id: 'box-map-50-95', key: 'boxMap5095', label: 'Box mAP@0.5:0.95', color: palette.boxValidation },
  { id: 'mask-map-50', key: 'maskMap50', label: 'Mask mAP@0.5', color: palette.maskTrain },
  { id: 'mask-map-50-95', key: 'maskMap5095', label: 'Mask mAP@0.5:0.95', color: palette.maskValidation },
];

const panels = [
  renderPanel({
    id: 'panel-a-loss',
    col: 0,
    title: '(a) Box and Mask Loss',
    yLabel: 'Loss',
    yDomain: [0, 1.8],
    yTicks: ticks(0, 1.8, 0.3),
    yFormat: (value) => value.toFixed(1),
    series: lossSeries,
    legend: { items: lossSeries, columns: 2, columnWidth: 158 },
    legendPosition: { x: 142, y: 52 },
    markerKeys: ['trainBox', 'valBox', 'trainMask', 'valMask'],
  }),
  renderPanel({
    id: 'panel-b-map',
    col: 1,
    title: '(b) Box and Mask mAP',
    yLabel: 'mAP (%)',
    yDomain: [40, 100],
    yTicks: ticks(40, 100, 10),
    yFormat: (value) => value.toFixed(0),
    series: mapSeries,
    legend: { items: mapSeries, columns: 2, columnWidth: 206 },
    legendPosition: { x: 50, y: 423 },
    markerKeys: ['boxMap5095', 'maskMap5095'],
    bestValueLabel: {
      key: 'maskMap5095',
      format: (value) => `Mask mAP@0.5:0.95 = ${value.toFixed(2)}%`,
    },
  }),
];

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" role="img" aria-labelledby="figure-title figure-description">
  <title id="figure-title">YOLO box loss, mask loss, and mask mAP training curves</title>
  <desc id="figure-description">Panel a compares training and validation box and mask losses. Panel b shows mask mAP at IoU 0.5 and averaged from 0.5 to 0.95. Epoch ${bestRecord.epoch} is marked as the best epoch.</desc>
  <rect id="background" width="100%" height="100%" fill="${palette.white}"/>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: ${palette.text}; }
    .panel-title { font-size: 24px; font-weight: 500; }
    .axis-label { font-size: 22px; font-weight: 400; }
    .tick-label { font-size: 18px; font-weight: 400; }
    .axis { stroke: ${palette.axis}; stroke-width: 2.2; shape-rendering: geometricPrecision; }
    .legend-box { fill: ${palette.white}; fill-opacity: 0.94; stroke: ${palette.border}; stroke-width: 1.5; }
    .legend-label { font-size: 16px; font-weight: 400; }
    .best-line { stroke: ${palette.best}; stroke-width: 1.8; }
    .best-label {
      font-size: 17px;
      font-weight: 500;
      fill: ${palette.best};
      stroke: ${palette.white};
      stroke-width: 7;
      stroke-linejoin: round;
      paint-order: stroke fill;
    }
  </style>
  ${panels.join('\n')}
</svg>
`;

fs.writeFileSync(`${outputBase}.svg`, svg, 'utf8');

async function writePreview() {
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
      return;
    }
  }
  await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .withMetadata({ density: 300 })
    .toFile(`${outputBase}_preview.png`);
}

writePreview()
  .then(() => {
    console.log(`Best epoch: ${bestRecord.epoch}`);
    console.log(`Mask mAP@0.5:0.95: ${bestRecord.maskMap5095.toFixed(5)}%`);
    console.log(`Generated ${outputBase}.svg`);
  })
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
