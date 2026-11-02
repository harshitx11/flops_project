const config = window.SENTIMENT_CONFIG || {};
const apiBaseUrl = (config.apiBaseUrl || '').replace(/\/$/, '');
const $ = (selector) => document.querySelector(selector);

const form = $('#analyze-form');
const review = $('#review');
const counter = $('#counter');
const submitButton = $('#submit-button');
const resultCard = $('#result');
const errorCard = $('#error');
const history = [];

const percent = (value) => `${(value * 100).toFixed(1)}%`;

function updateCounter() {
  counter.textContent = `${review.value.length} / ${review.maxLength}`;
}

function renderTerms(list, terms) {
  list.replaceChildren();
  if (!terms.length) {
    const item = document.createElement('li');
    item.className = 'none';
    item.textContent = 'No known words';
    list.append(item);
    return;
  }
  terms.forEach(({ term, weight }) => {
    const item = document.createElement('li');
    item.textContent = term;
    item.title = `Contribution ${weight.toFixed(3)}`;
    list.append(item);
  });
}

function renderHistory() {
  const list = $('#history');
  list.replaceChildren();
  if (!history.length) {
    const empty = document.createElement('li');
    empty.className = 'empty';
    empty.textContent = 'No reviews analyzed yet.';
    list.append(empty);
    return;
  }
  history.forEach((entry) => {
    const item = document.createElement('li');
    const tag = document.createElement('span');
    tag.className = `tag ${entry.predicted_class}`;
    tag.textContent = entry.label;
    const text = document.createElement('button');
    text.type = 'button';
    text.className = 'history-text';
    text.textContent = entry.text;
    text.addEventListener('click', () => {
      review.value = entry.text;
      updateCounter();
      review.focus();
    });
    item.append(tag, text);
    list.append(item);
  });
}

function showResult(payload) {
  const positive = payload.predicted_class === 'positive';
  const probability = payload.positive_probability;
  const confidence = positive ? probability : 1 - probability;
  resultCard.className = `card result ${payload.predicted_class}`;
  $('#verdict-icon').textContent = positive ? '😊' : '😞';
  $('#result-title').textContent = `${payload.label} sentiment`;
  $('#confidence').textContent = `${percent(confidence)} confident`;
  $('#meter-marker').style.left = `${probability * 100}%`;
  $('#meter-text').textContent = `Positive probability ${percent(probability)} · model v${payload.model_version}`;
  renderTerms($('#positive-terms'), payload.positive_terms);
  renderTerms($('#negative-terms'), payload.negative_terms);
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
  errorCard.textContent = message;
  errorCard.hidden = false;
}

async function loadStatus() {
  const status = $('#api-status');
  try {
    const response = await fetch(`${apiBaseUrl}/health`);
    const health = await response.json();
    const ready = response.ok && health.model_loaded;
    status.textContent = ready ? `Model v${health.model_version} ready` : 'Model unavailable';
    status.classList.toggle('ready', ready);
  } catch {
    status.textContent = 'API unreachable';
  }
}

async function loadModelCard() {
  const stats = $('#model-stats');
  try {
    const response = await fetch(`${apiBaseUrl}/model`);
    if (!response.ok) throw new Error();
    const card = await response.json();
    const rows = [
      ['Accuracy', percent(card.metrics.accuracy)],
      ['Macro F1', percent(card.metrics.macro_f1)],
      ['Train / test', `${card.dataset.train_records} / ${card.dataset.test_records}`],
      ['Trained', new Date(card.trained_at).toLocaleDateString()],
    ];
    stats.replaceChildren(...rows.map(([term, value]) => {
      const row = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = term;
      dd.textContent = value;
      row.append(dt, dd);
      return row;
    }));
  } catch {
    stats.innerHTML = '<div><dt>Status</dt><dd>Unavailable</dd></div>';
  }
}

review.addEventListener('input', updateCounter);
review.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) form.requestSubmit();
});
$('#clear-button').addEventListener('click', () => {
  review.value = '';
  resultCard.hidden = true;
  errorCard.hidden = true;
  updateCounter();
  review.focus();
});
document.querySelectorAll('.chip').forEach((chip) => chip.addEventListener('click', () => {
  review.value = chip.dataset.text;
  updateCounter();
  form.requestSubmit();
}));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = review.value.trim();
  if (!text) return;
  submitButton.disabled = true;
  submitButton.textContent = 'Analyzing…';
  errorCard.hidden = true;
  try {
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(payload.detail) ? payload.detail[0].msg : payload.detail;
      throw new Error(detail || 'The API could not process this review.');
    }
    showResult(payload);
    history.unshift({ text, label: payload.label, predicted_class: payload.predicted_class });
    history.splice(5);
    renderHistory();
  } catch (error) {
    resultCard.hidden = true;
    showError(error instanceof TypeError ? 'Could not reach the API. Is the server running?' : error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Analyze sentiment';
  }
});

updateCounter();
loadStatus();
loadModelCard();
