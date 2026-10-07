<script setup>
import { computed, onMounted, ref } from 'vue'

const data = ref(null)
const loading = ref(true)
const error = ref('')
const refreshing = ref(false)
const loadedAt = ref(null)

async function loadDashboard(manual = false) {
  if (manual) refreshing.value = true
  else loading.value = true
  error.value = ''
  try {
    const response = await fetch('/api/dashboard', { cache: 'no-store' })
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`)
    data.value = await response.json()
    loadedAt.value = new Date()
  } catch (cause) {
    error.value = cause.message || 'Could not load dashboard data.'
  } finally {
    loading.value = false
    refreshing.value = false
  }
}

function money(value, currency) {
  if (value === null || value === undefined) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: currency || 'USD', maximumFractionDigits: 2,
  }).format(value)
}

function percent(value) {
  if (value === null || value === undefined) return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`
}

function chartPoints(history) {
  if (!history || history.length < 2) return ''
  const values = history.map((item) => item.close)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  return values.map((value, index) => {
    const x = 4 + (index / (values.length - 1)) * 292
    const y = 88 - ((value - min) / range) * 72
    return `${x},${y}`
  }).join(' ')
}

function chartArea(history) {
  const points = chartPoints(history)
  if (!points) return ''
  const first = points.split(' ')[0].split(',')[0]
  const last = points.split(' ').at(-1).split(',')[0]
  return `${first},92 ${points} ${last},92`
}

function asOf(value) {
  if (!value) return 'No quote available'
  return `Latest session · ${new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
}

function published(value) {
  if (!value) return 'Time unavailable'
  return new Date(value).toLocaleString('en-US', {
    timeZone: 'Asia/Taipei', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

const stockCount = computed(() => data.value?.stocks?.length || 0)
const newsCount = computed(() => data.value?.news_count ?? data.value?.news?.length ?? 0)

function formatTopic(topic) {
  return String(topic).replaceAll('_', ' ')
}

onMounted(() => loadDashboard())
</script>

<template>
  <div class="app-shell">
    <aside class="rail">
      <a class="brand-mark" href="#top" aria-label="StockProjectX home">S<span>.</span></a>
      <div class="rail-rule"></div>
      <a class="rail-link active" href="#overview" aria-label="Overview" title="Overview">
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="3" width="8" height="5" rx="2"/><rect x="13" y="10" width="8" height="11" rx="2"/><rect x="3" y="13" width="8" height="8" rx="2"/></svg>
      </a>
      <a class="rail-link" href="#news" aria-label="News" title="News">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h14a2 2 0 0 1 2 2v13H5a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3Z"/><path d="M7 8h9M7 12h10M7 16h5"/></svg>
      </a>
      <div class="rail-bottom"><span class="online-dot"></span></div>
    </aside>

    <main id="top" class="main-content">
      <header class="topbar">
        <div class="crumb"><span>WORKSPACE</span><b>/</b><strong>Market brief</strong></div>
        <div class="top-actions">
          <div class="time-chip"><span class="online-dot"></span> TAIPEI SESSION <b>08:00</b></div>
          <button class="refresh-button" :disabled="refreshing" @click="loadDashboard(true)">
            <svg :class="{ spin: refreshing }" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.6 9a7 7 0 0 1 11.7-2L20 12M4 12l2.7 5a7 7 0 0 0 11.7-2"/></svg>
            {{ refreshing ? 'Updating' : 'Refresh view' }}
          </button>
        </div>
      </header>

      <section id="overview" class="hero-row">
        <div>
          <div class="eyebrow"><span class="eyebrow-line"></span> DAILY MARKET INTELLIGENCE</div>
          <h1>Good morning<span class="period">.</span></h1>
          <p class="hero-copy">Your watchlist, the latest moves, and the stories behind them.</p>
        </div>
        <div class="window-card">
          <div class="window-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="16" rx="3"/><path d="M16 3v4M8 3v4M3 10h18"/></svg></div>
          <div><span class="window-label">NEWS WINDOW · TAIPEI TIME</span><strong>{{ data?.window?.label || '48 hours ending at 08:00 Taiwan time' }}</strong></div>
          <span class="window-status">DAILY RUN</span>
        </div>
      </section>

      <div v-if="error" class="error-banner"><strong>Dashboard data could not load.</strong> {{ error }} <button @click="loadDashboard(true)">Try again</button></div>

      <section class="section-heading">
        <div><span class="section-index">01</span><h2>Market pulse</h2><span class="section-count">{{ stockCount }} SECURITIES</span></div>
        <span class="data-note"><i></i> Market data · latest available</span>
      </section>

      <section class="stock-grid" aria-label="Stock prices">
        <article v-for="stock in data?.stocks || []" :key="stock.id" class="stock-card" :class="{ 'stock-down': stock.change_percent < 0 }">
          <div class="stock-card-top">
            <div class="stock-identity"><span class="company-icon" :class="stock.id === 'tesla' ? 'tesla-icon' : ['tsmc_adr', 'tsmc_taiwan'].includes(stock.id) ? 'tsmc-icon' : 'generic-icon'">{{ stock.display_symbol.slice(0, 2) }}</span><div><span class="stock-market">{{ stock.market }} MARKET</span><h3>{{ stock.display_symbol }} <small>{{ stock.name }}</small></h3></div></div>
            <span class="quote-tag" :class="stock.status">{{ stock.status === 'available' ? '● LATEST QUOTE' : '○ NO DATA' }}</span>
          </div>
          <div class="stock-main">
            <div><div class="stock-price">{{ money(stock.price, stock.currency) }}</div><div class="stock-updated">{{ asOf(stock.as_of) }}</div></div>
            <div class="change-pill" :class="stock.change_percent >= 0 ? 'positive' : 'negative'">
              <span>{{ stock.change_percent >= 0 ? '↗' : '↘' }}</span>{{ percent(stock.change_percent) }}
            </div>
          </div>
          <div class="chart-wrap">
            <svg class="stock-chart" viewBox="0 0 300 96" preserveAspectRatio="none" role="img" :aria-label="`${stock.display_symbol} one month price trend`">
              <defs><linearGradient :id="`fill-${stock.id}`" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" :class="stock.change_percent >= 0 ? 'fill-green' : 'fill-red'"/><stop offset="100%" class="fill-clear"/></linearGradient></defs>
              <path v-if="chartArea(stock.history)" :d="`M ${chartArea(stock.history)} Z`" :fill="`url(#fill-${stock.id})`" />
              <polyline v-if="chartPoints(stock.history)" :points="chartPoints(stock.history)" :class="stock.change_percent >= 0 ? 'line-green' : 'line-red'" />
              <text v-if="!chartPoints(stock.history)" x="150" y="50" text-anchor="middle">Chart data unavailable</text>
            </svg>
            <div class="chart-axis"><span>1 MONTH</span><span>DAILY CLOSE</span><span>NOW</span></div>
          </div>
        </article>
        <div v-if="loading && !data" class="loading-card">Loading market data…</div>
      </section>
      <p class="market-footnote">Publicly traded watchlist securities are shown above. SpaceX, OpenAI, and Anthropic are private companies without public stock quotes.</p>

      <section class="section-heading news-heading" id="news">
        <div><span class="section-index">02</span><h2>News desk</h2><span class="section-count">{{ newsCount }} STORIES</span></div>
        <span class="data-note">Company and global-topic highlights · newest first</span>
      </section>

      <section class="content-grid">
        <div class="news-panel panel">
          <div class="panel-head"><div><h3>Top stories</h3><p>{{ data?.news_mode === 'archive' ? 'Latest saved stories · outside this window' : 'Company news and global market topics · source links included' }}</p></div><span class="live-label" :class="{ archived: data?.news_mode === 'archive' }"><i></i> {{ data?.news_mode === 'archive' ? 'ARCHIVE PREVIEW' : '48H WINDOW' }}</span></div>
          <div v-if="loading && !data" class="empty-state">Loading the latest stories…</div>
          <div v-else-if="!data?.news?.length" class="empty-state"><span class="empty-icon">◷</span><strong>No matching stories in this window</strong><p>New stories will appear after the 08:00 Taiwan-time collection completes.</p></div>
          <article v-for="(article, index) in data?.news || []" :key="article.url" class="news-item">
            <div class="news-number">0{{ index + 1 }}</div>
            <div class="news-body">
              <div class="news-meta"><span class="source-name">{{ article.source?.name || 'NEWS SOURCE' }}</span><span class="meta-dot"></span><span>{{ published(article.published_at) }} TST</span><span class="lang-chip">{{ article.language }}</span><span v-if="article.topics?.length" class="global-chip">GLOBAL TOPIC</span></div>
              <a class="news-title" :href="article.url" target="_blank" rel="noreferrer">{{ article.display_title }}</a>
              <p class="news-summary">{{ article.display_summary || 'Summary is not available yet.' }}</p>
              <div class="news-footer"><div class="entity-tags"><span v-for="entity in (article.entities || []).slice(0, 3)" :key="entity.id">{{ entity.name }}</span><span v-for="topic in (article.topics || [])" :key="topic" class="topic-tag">{{ formatTopic(topic) }}</span></div><a class="source-link" :href="article.url" target="_blank" rel="noreferrer">VIEW SOURCE <span>↗</span></a></div>
            </div>
          </article>
        </div>

        <aside class="right-column">
          <section class="panel impact-panel">
            <div class="panel-head"><div><span class="panel-kicker">03 · SIGNAL REVIEW</span><h3>What could move?</h3></div><span class="beta-chip">{{ data?.impact_analysis?.status === 'unavailable' ? 'MODEL UNAVAILABLE' : data?.impact_analysis?.status === 'no_articles' ? 'NO CURRENT STORIES' : data?.impact_analysis?.status === 'fallback_sentiment' ? 'FINBERT FALLBACK' : 'LOCAL HF MODEL' }}</span></div>
            <p class="impact-intro">Local analysis of saved company and global-topic news, mapped to watchlist securities. Cues are hypotheses, not stock price forecasts.</p>
            <div v-if="data?.impact_analysis?.evidence_brief?.summary" class="evidence-brief"><strong>PRE-ANALYSIS EVIDENCE BRIEF</strong><p>{{ data.impact_analysis.evidence_brief.summary }}</p></div>
            <div class="impact-row upside"><div class="impact-icon">↗</div><div class="impact-content"><strong>Potential upside cues</strong>
              <template v-if="data?.impact_analysis?.upside?.length"><div v-for="(signal, index) in data.impact_analysis.upside" :key="`${signal.url}-${index}`" class="signal-item"><p>{{ signal.title }}</p><small>Possible upside · model confidence {{ (signal.score * 100).toFixed(0) }}%<span v-if="signal.evidence_priority_score != null"> · evidence priority {{ signal.evidence_priority_score }}/100</span><span v-if="signal.symbols?.length"> · {{ signal.symbols.join(', ') }}</span></small><p v-if="signal.reason" class="signal-summary">{{ signal.reason }}</p><p v-else-if="signal.summary" class="signal-summary">{{ signal.summary }}</p><a :href="signal.url" target="_blank" rel="noreferrer">SOURCE ↗</a></div></template>
              <p v-else class="no-signal">{{ data?.impact_analysis?.status === 'unavailable' ? 'The local sentiment model could not run.' : 'No positive sentiment cues in the stories shown.' }}</p>
            </div></div>
            <div class="impact-row downside"><div class="impact-icon">↘</div><div class="impact-content"><strong>Potential downside cues</strong>
              <template v-if="data?.impact_analysis?.downside?.length"><div v-for="(signal, index) in data.impact_analysis.downside" :key="`${signal.url}-${index}`" class="signal-item"><p>{{ signal.title }}</p><small>Possible downside · model confidence {{ (signal.score * 100).toFixed(0) }}%<span v-if="signal.evidence_priority_score != null"> · evidence priority {{ signal.evidence_priority_score }}/100</span><span v-if="signal.symbols?.length"> · {{ signal.symbols.join(', ') }}</span></small><p v-if="signal.reason" class="signal-summary">{{ signal.reason }}</p><p v-else-if="signal.summary" class="signal-summary">{{ signal.summary }}</p><a :href="signal.url" target="_blank" rel="noreferrer">SOURCE ↗</a></div></template>
              <p v-else class="no-signal">{{ data?.impact_analysis?.status === 'unavailable' ? 'The local sentiment model could not run.' : 'No negative sentiment cues in the stories shown.' }}</p>
            </div></div>
            <p v-if="data?.impact_analysis?.neutral?.length" class="neutral-note">{{ data.impact_analysis.neutral.length }} {{ data.impact_analysis.neutral.length === 1 ? 'story was' : 'stories were' }} labeled neutral and are not shown as upside or downside cues.</p>
            <div class="analysis-note"><span>i</span>{{ data?.impact_analysis?.message || 'Signals require review against original reporting.' }}</div>
          </section>

          <section class="takeaway-card">
            <div class="takeaway-top"><span class="takeaway-icon">✳</span><span class="panel-kicker">04 · DAILY TAKEAWAY</span></div>
            <h3>Today’s investor note</h3>
            <p>{{ data?.daily_takeaway || 'Review verified reporting and market data before making an investment decision.' }}</p>
            <div class="takeaway-foot"><span>DATA-LED CONTEXT</span><span>NOT PERSONALIZED ADVICE</span></div>
          </section>

          <div class="data-footnote"><span class="footnote-mark">↗</span><p>Quotes use the latest available market data and can be delayed. Stories link to their original publishers.</p></div>
        </aside>
      </section>

      <section class="validation-card" aria-label="Prediction validation history">
        <div class="validation-copy">
          <span class="panel-kicker">05 · TRACK RECORD</span>
          <h2>Prediction validation</h2>
          <p>Scores original frozen calls against their exact target-session close. Each stock needs 30 completed observations and 40 usable training samples. Confidence below 60% produces an uncertain result. Refreshed news cues may differ from the original tracked calls.</p>
        </div>
        <div class="validation-metrics">
          <div><span>TRACKING DAYS</span><strong>{{ data?.validation?.days_running ?? 0 }}</strong></div>
          <div><span>LSTM ACCURACY</span><strong>{{ data?.validation?.accuracy_percent == null ? '—' : `${data.validation.accuracy_percent.toFixed(1)}%` }}</strong></div>
          <div><span>EVALUATED CALLS</span><strong>{{ data?.validation?.correct_calls ?? 0 }} / {{ data?.validation?.evaluated_calls ?? 0 }}</strong></div>
          <div><span>MIN STOCK OBSERVATIONS</span><strong>{{ data?.validation?.lstm_training_days ?? 0 }} / {{ data?.validation?.lstm_required_days ?? 30 }}</strong></div>
          <div><span>LSTM COVERAGE</span><strong>{{ data?.validation?.coverage_percent == null ? '—' : `${data.validation.coverage_percent.toFixed(1)}%` }}</strong></div>
          <div><span>PENDING LSTM CALLS</span><strong>{{ data?.validation?.pending_calls ?? 0 }}</strong></div>
          <div><span>BRIER SCORE</span><strong>{{ data?.validation?.brier_score == null ? '—' : data.validation.brier_score.toFixed(3) }}</strong></div>
          <span class="beta-chip">{{ data?.validation?.lstm_status === 'available' ? 'LSTM ACTIVE' : `LSTM ${(data?.validation?.lstm_status || 'warming_up').replaceAll('_', ' ').toUpperCase()}` }}</span>
        </div>
        <div class="validation-details">
          <p>Coverage is the share of matured forecasts that made an up/down call. Brier score measures raw probability error (lower is better); confidence is uncalibrated. Flat closes count as incorrect for directional calls. Splits are excluded.</p>
          <div class="validation-table-wrap">
            <table class="validation-table"><thead><tr><th>Model</th><th>Overall accuracy</th><th>Coverage</th><th>Accuracy on matched LSTM calls</th></tr></thead><tbody>
              <tr v-for="(metric, model) in (data?.validation?.models || {})" :key="model"><td>{{ model.replaceAll('_', ' ') }}</td><td>{{ metric.accuracy_percent == null ? '—' : `${metric.accuracy_percent.toFixed(1)}%` }} ({{ metric.evaluated_calls }})</td><td>{{ metric.coverage_percent == null ? '—' : `${metric.coverage_percent.toFixed(1)}%` }}</td><td>{{ metric.paired_with_lstm?.accuracy_percent == null ? '—' : `${metric.paired_with_lstm.accuracy_percent.toFixed(1)}%` }} ({{ metric.paired_with_lstm?.evaluated_calls ?? 0 }})</td></tr>
            </tbody></table>
          </div>
          <div class="training-coverage"><div v-for="stock in (data?.validation?.training_coverage || [])" :key="stock.security_id"><strong>{{ stock.symbol || stock.security_id }}</strong><span>{{ stock.observations }}/{{ stock.required_observations }} observations · {{ stock.training_samples }}/{{ stock.required_samples }} samples · {{ stock.status.replaceAll('_', ' ') }}</span></div></div>
          <p v-if="data?.validation?.reference_price_rejections?.length">Forecasts withheld for {{ data.validation.reference_price_rejections.map(item => item.security_id).join(', ') }}: a valid completed reference close was unavailable.</p>
          <p>Matched comparisons use the same stock, target session, and issuance batch. Earlier legacy scores are preserved in history and excluded from these metrics.</p>
        </div>
      </section>

      <footer class="footer"><span>STOCKPROJECTX <b>·</b> MULTILINGUAL MARKET BRIEF</span><span v-if="loadedAt">VIEW UPDATED {{ loadedAt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) }}</span><span v-else>LOCAL DASHBOARD</span></footer>
    </main>
  </div>
</template>
