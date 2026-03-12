const METRICS_URL = './results/eval_metrics.json';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch(METRICS_URL);
        if (!resp.ok) throw new Error('Metrics not found');
        const data = await resp.json();
        renderMetrics(data);
    } catch (e) {
        console.error('Failed to load metrics:', e);
        document.getElementById('metrics-status').textContent = 'Metrics not yet available — run the Colab notebook first.';
    }
});

function renderMetrics(data) {
    const ev = data.evaluation;
    const tr = data.training;

    // Hero metrics
    setText('json-improvement', `${((ev.dpo.valid_json_rate - ev.base.valid_json_rate) * 100).toFixed(0)}%`);
    setText('f1-improvement', `${((ev.dpo.avg_key_f1 - ev.base.avg_key_f1) * 100).toFixed(1)}pts`);
    setText('val-improvement', `${((ev.dpo.avg_value_accuracy - ev.base.avg_value_accuracy) * 100).toFixed(1)}pts`);
    setText('trainable-pct', `${tr.sft.trainable_params_pct}%`);

    // Summary cards
    setText('base-json', `${(ev.base.valid_json_rate * 100).toFixed(1)}%`);
    setText('sft-json', `${(ev.sft.valid_json_rate * 100).toFixed(1)}%`);
    setText('dpo-json', `${(ev.dpo.valid_json_rate * 100).toFixed(1)}%`);

    setText('base-f1', ev.base.avg_key_f1.toFixed(3));
    setText('sft-f1', ev.sft.avg_key_f1.toFixed(3));
    setText('dpo-f1', ev.dpo.avg_key_f1.toFixed(3));

    setText('base-val', ev.base.avg_value_accuracy.toFixed(3));
    setText('sft-val', ev.sft.avg_value_accuracy.toFixed(3));
    setText('dpo-val', ev.dpo.avg_value_accuracy.toFixed(3));

    setText('base-lat', `${ev.base.avg_latency_s.toFixed(2)}s`);
    setText('sft-lat', `${ev.sft.avg_latency_s.toFixed(2)}s`);
    setText('dpo-lat', `${ev.dpo.avg_latency_s.toFixed(2)}s`);

    // Comparison bars
    renderComparisonBar('json-bars', 'Valid JSON Rate', [
        { label: 'Base', value: ev.base.valid_json_rate * 100, cls: 'base' },
        { label: 'SFT', value: ev.sft.valid_json_rate * 100, cls: 'sft' },
        { label: 'SFT+DPO', value: ev.dpo.valid_json_rate * 100, cls: 'dpo' },
    ], '%');

    renderComparisonBar('f1-bars', 'Key Extraction F1', [
        { label: 'Base', value: ev.base.avg_key_f1 * 100, cls: 'base' },
        { label: 'SFT', value: ev.sft.avg_key_f1 * 100, cls: 'sft' },
        { label: 'SFT+DPO', value: ev.dpo.avg_key_f1 * 100, cls: 'dpo' },
    ], '%');

    renderComparisonBar('val-bars', 'Value Accuracy', [
        { label: 'Base', value: ev.base.avg_value_accuracy * 100, cls: 'base' },
        { label: 'SFT', value: ev.sft.avg_value_accuracy * 100, cls: 'sft' },
        { label: 'SFT+DPO', value: ev.dpo.avg_value_accuracy * 100, cls: 'dpo' },
    ], '%');

    renderComparisonBar('latency-bars', 'Inference Latency', [
        { label: 'Base', value: ev.base.avg_latency_s, cls: 'base' },
        { label: 'SFT', value: ev.sft.avg_latency_s, cls: 'sft' },
        { label: 'SFT+DPO', value: ev.dpo.avg_latency_s, cls: 'dpo' },
    ], 's', true);

    // Training summary
    setText('sft-train-loss', tr.sft.train_loss.toFixed(3));
    setText('sft-eval-loss', tr.sft.eval_loss.toFixed(3));
    setText('dpo-train-loss', tr.dpo.train_loss.toFixed(3));
    setText('dpo-eval-loss', tr.dpo.eval_loss.toFixed(3));

    // Results table
    renderResultsTable(ev);

    // Timestamp
    const ts = document.getElementById('metrics-timestamp');
    if (ts) {
        const isPlaceholder = data.note && data.note.includes('Placeholder');
        ts.textContent = isPlaceholder
            ? 'Showing placeholder metrics — run Colab notebook for real results'
            : `Results from: ${new Date(data.timestamp).toLocaleString()}`;
        ts.style.color = isPlaceholder ? 'var(--warning)' : 'var(--text-muted)';
    }

    const status = document.getElementById('metrics-status');
    if (status) status.style.display = 'none';
}

function renderComparisonBar(containerId, title, items, unit, lowerIsBetter = false) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const maxVal = Math.max(...items.map(i => i.value));
    let html = `<h3>${title}</h3>`;

    items.forEach(item => {
        const pct = maxVal > 0 ? (item.value / maxVal) * 100 : 0;
        const display = unit === '%' ? `${item.value.toFixed(1)}%` : `${item.value.toFixed(2)}${unit}`;
        html += `
        <div class="bar-row">
            <span class="bar-label">${item.label}</span>
            <div class="bar-track">
                <div class="bar-fill ${item.cls}" style="width: ${Math.max(pct, 3)}%">
                    <span>${display}</span>
                </div>
            </div>
        </div>`;
    });

    container.innerHTML = html;
}

function renderResultsTable(ev) {
    const wrapper = document.getElementById('results-table');
    if (!wrapper) return;

    const metrics = ['valid_json_rate', 'avg_key_f1', 'avg_value_accuracy', 'avg_latency_s', 'tokens_per_second'];
    const labels = ['Valid JSON Rate', 'Key F1 Score', 'Value Accuracy', 'Avg Latency', 'Tokens/sec'];

    let html = `<table class="results-table">
        <thead><tr><th>Metric</th><th>Base</th><th>SFT</th><th>SFT + DPO</th><th>Improvement</th></tr></thead><tbody>`;

    metrics.forEach((m, i) => {
        const base = ev.base[m];
        const sft = ev.sft[m];
        const dpo = ev.dpo[m];
        const isLatency = m === 'avg_latency_s';
        const fmt = m === 'valid_json_rate' ? v => `${(v*100).toFixed(1)}%` :
                    m === 'avg_latency_s' ? v => `${v.toFixed(2)}s` :
                    m === 'tokens_per_second' ? v => `${v.toFixed(1)}` :
                    v => v.toFixed(3);

        const diff = isLatency ? base - dpo : dpo - base;
        const pctChange = base !== 0 ? (diff / Math.abs(base) * 100) : 0;
        const isPositive = isLatency ? diff > 0 : diff > 0;
        const arrow = isPositive ? '+' : '';
        const cls = isPositive ? 'positive' : 'neutral';

        html += `<tr>
            <td style="font-family: var(--font-sans); font-weight: 500;">${labels[i]}</td>
            <td>${fmt(base)}</td>
            <td>${fmt(sft)}</td>
            <td>${fmt(dpo)}</td>
            <td><span class="improvement ${cls}">${arrow}${pctChange.toFixed(1)}%</span></td>
        </tr>`;
    });

    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
