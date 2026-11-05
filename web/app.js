// Demo Bank assistant: sign-in, chat with live progress, confirmation slips with security codes,
// and the "What happened" trace. An answer is started with POST /api/chat and then checked on
// (GET /api/turns/{id}) every second until it's ready, so no request is held open for long.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const els = {
    signIn: $("sign-in"), chat: $("chat"), customers: $("customers"), owners: $("owners"), who: $("who"),
    whoName: $("who-name"), log: $("log"), empty: $("empty"), form: $("composer"), input: $("message"),
    send: $("send"), trace: $("trace"), signOut: $("sign-out"), resetBank: $("reset-bank"),
    slipTemplate: $("slip-template"), notice: $("notice"), visitorForm: $("visitor-form"),
    visitorEmail: $("visitor-email"), visitorName: $("visitor-name"), visitorNote: $("visitor-note"),
    visitorGo: $("visitor-go"), visitorTerms: $("visitor-terms"), publicSignin: $("public-signin"),
    googleButton: $("google-button"), orDivider: $("or-divider"), emailForm: $("email-form"),
    signinEmail: $("signin-email"), emailGo: $("email-go"), codeForm: $("code-form"), signinCode: $("signin-code"),
    signinCodeLabel: $("signin-code-label"), codeGo: $("code-go"), codeRestart: $("code-restart"),
    publicNote: $("public-note"), choose: $("choose"), chooseTitle: $("choose-title"),
    chooseCustomers: $("choose-customers"), chooseNote: $("choose-note"), switchCustomer: $("switch-customer"),
    convos: $("convos"), convoNew: $("convo-new"), menu: $("menu-button"), panel: $("convo-panel"),
    scrim: $("scrim"), convoClose: $("convo-close"), conversation: document.querySelector(".conversation"),
    traceAside: document.querySelector(".trace"),
  };
  let signinChallenge = null;
  const SIGNIN_ERRORS = {
    cancelled: "Google sign-in was cancelled.",
    expired: "That sign-in took too long, or was started in another browser. Please try again.",
    state: "That sign-in link wasn't started here. Please try again.",
    google: "Google sign-in didn't work just now. Please try again, or use an email code.",
    unverified: "Your Google account's email address isn't verified, so it can't be used here.",
    invite: "The demo is invite-only for now, and that address isn't on the list.",
    signups: "Too many new demo banks from your network in the last hour. Please try again later.",
  };
  let me = null;
  let config = { customers: [], visitors: { bank_hours: 24, daily_messages: 30 } };
  let threadId = null;
  let busy = false;

  // ---------- storage (per-viewer convenience only; the server holds the real state) ----------
  const store = {
    get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* private mode */ } },
    del(key) { try { sessionStorage.removeItem(key); } catch { /* private mode */ } },
  };
  const threadKey = () => `thread-${me && me.customer}`;

  // ---------- API ----------
  class ApiError extends Error {
    constructor(message, status, data) { super(message); this.status = status; this.data = data; }
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
    });
    let data = {};
    try { data = await res.json(); } catch { /* empty body */ }
    if (res.status === 401 && path !== "/api/login") {
      showSignIn(data.detail);
      throw new ApiError("Please sign in.", 401, data);
    }
    if (!res.ok && !(res.status === 409 && data.status)) {
      const detail = typeof data.detail === "string" ? data.detail : null;
      throw new ApiError(detail || `Something went wrong (${res.status}).`, res.status, data);
    }
    return { ...data, httpStatus: res.status };
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Check back on an answer until it's ready, showing each step as it happens. Each check is a
  // "long" one: the server holds it for up to 20 seconds and replies as soon as there's news.
  async function follow(turnId, progressEl) {
    const started = Date.now();
    let seen = 0;
    while (Date.now() - started < 5 * 60 * 1000) {
      const asked = Date.now();
      let data;
      try {
        data = await api(`/api/turns/${encodeURIComponent(turnId)}?wait=20&seen=${seen}`);
      } catch (err) {
        if (err.status === 401 || err.status === 404 || (err.data && err.data.closed)) throw err;
        await sleep(1000);
        continue;   // a network blip: keep checking
      }
      if (data.status !== "running") return data;
      seen = (data.steps || []).length;
      if (progressEl) showProgress(progressEl, data.steps);
      if (Date.now() - asked < 300) await sleep(500);   // an older server that doesn't hold checks
    }
    throw new Error("This is taking longer than it should. Reload the page in a minute to see the answer.");
  }

  // ---------- safe markdown (escape first, then a small subset) ----------
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  function markdown(text) {
    const lines = String(text || "").replace(/\r/g, "").split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*\|/.test(line)) {                       // table
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
        const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const isSep = (r) => cells(r).every((c) => /^:?-{2,}:?$/.test(c));
        const [head, ...rest] = rows.filter((r) => !isSep(r));
        if (head) {
          out.push("<table><thead><tr>" + cells(head).map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>"
            + rest.map((r) => "<tr>" + cells(r).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("")
            + "</tbody></table>");
        }
        continue;
      }
      if (/^\s*[-*]\s+/.test(line)) {                  // bullet list
        const items = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*]\s+/, ""));
        out.push("<ul>" + items.map((t) => `<li>${inline(t)}</li>`).join("") + "</ul>");
        continue;
      }
      if (/^\s*\d+[.)]\s+/.test(line)) {               // numbered list
        const items = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ""));
        out.push("<ol>" + items.map((t) => `<li>${inline(t)}</li>`).join("") + "</ol>");
        continue;
      }
      if (!line.trim()) { i++; continue; }
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*(\||[-*]\s|\d+[.)]\s)/.test(lines[i])) para.push(lines[i++]);
      out.push(`<p>${para.map((p) => inline(p.replace(/^#+\s*/, ""))).join("<br>")}</p>`);
    }
    return out.join("");
  }

  // ---------- charts: drawn from the bank's own figures (checked by api/charts.py first) ----------
  // Forms: column (one set of figures; several stand side by side), stacked, line, bar (ranked,
  // horizontal) and donut. One set of figures needs no legend (the title names it); several get one.
  // Every value is also in the table under the chart, and on hover or keyboard focus. Names go in
  // with textContent only. The colours are checked for colour-blind separation (adjacent pairs; the
  // first three against each other, for lines that cross); "Other" is a quiet grey.
  const SVG = "http://www.w3.org/2000/svg";
  const svgEl = (name, attrs = {}) => {
    const n = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const COLORS = ["#008C7A", "#EB6834", "#6250D6", "#EDA100", "#2A78D6", "#E87BA4"];
  const OTHER = "#A3B1B4";
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const LONG_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December"];
  const monthOf = (key) => { const [y, m] = key.split("-").map(Number); return { y, m: m - 1 }; };
  const dayText = (iso) => { const [y, m, d] = iso.split("-").map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; };
  const datesIn = (text) => String(text).replace(/\d{4}-\d{2}-\d{2}/g, dayText);
  const isOther = (name) => String(name).toLowerCase() === "other";

  function money(value, currency, compact = false) {
    try {
      return new Intl.NumberFormat("en-GB", compact
        ? { style: "currency", currency, notation: "compact", maximumFractionDigits: 1 }
        : { style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
    } catch { return `${value.toFixed(2)} ${currency}`; }
  }

  let measurer = null;
  function textWidth(text, el, size = 12, weight = 400) {   // how wide a label is in the page's font
    measurer = measurer || document.createElement("canvas").getContext("2d");
    measurer.font = `${weight} ${size}px ${getComputedStyle(el).fontFamily}`;
    return measurer.measureText(text).width;
  }

  function fitText(text, width, el) {   // shorten with … rather than overflow
    if (textWidth(text, el) <= width) return text;
    let s = text;
    while (s.length > 1 && textWidth(`${s}…`, el) > width) s = s.slice(0, -1);
    return `${s.trimEnd()}…`;
  }

  function niceTicks(max) {
    if (!(max > 0)) return [0, 10];
    const rough = max / 4;
    const mag = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= rough);
    const ticks = [];
    for (let v = 0; v < max + step * 0.999; v += step) ticks.push(Math.round(v * 100) / 100);
    if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);
    return ticks;
  }

  // A column with a rounded top (4px) and a square foot on the baseline.
  function colPath(x0, x1, yb, yt, round = true) {
    const r = round ? Math.min(4, yb - yt, (x1 - x0) / 2) : 0;
    return `M${x0},${yb}V${yt + r}Q${x0},${yt} ${x0 + r},${yt}H${x1 - r}Q${x1},${yt} ${x1},${yt + r}V${yb}Z`;
  }

  function renderChart(chart) {
    const cur = chart.currency || "GBP";
    const labels = chart.labels || [];
    const series = chart.series || [];
    const form = chart.form;
    const byMonth = chart.axis === "month";
    const byDay = chart.axis === "day";             // labels are dates: 2026-10-16
    const byDom = chart.axis === "dayofmonth";      // labels are days of the month: "16"
    const timeAxis = byMonth || byDay || byDom;
    const running = byDom && chart.running === true; // running totals (month to date): no sums of points
    const years = new Set(byMonth ? labels.map((k) => monthOf(k).y) : []);
    const spanMonths = new Set(byDay ? labels.map((k) => k.slice(0, 7)) : []);
    const short = (k) => {
      if (byDay) { const [, m, d] = k.split("-").map(Number); return spanMonths.size > 1 ? `${d} ${MONTHS[m - 1]}` : String(d); }
      if (!byMonth) return k;
      const { y, m } = monthOf(k);
      return years.size > 1 ? `${MONTHS[m]} ${String(y).slice(2)}` : MONTHS[m];
    };
    const long = (k) => {
      if (byDay) return dayText(k);
      if (byDom) return `Day ${k}`;
      if (!byMonth) return k;
      const { y, m } = monthOf(k); return `${LONG_MONTHS[m]} ${y}`;
    };
    const notes = chart.notes || {};
    const flag = (k) => (notes[k] ? "*" : "");
    const combo = form === "combo";   // columns for each day + running-total lines (this month solid, last month dashed)
    const LAST_MONTH = "#6E8085";      // a quiet grey for last month's dashed line: a reference, not the story
    const seriesColor = (j) => (combo ? (series[j].kind === "dashed" ? LAST_MONTH : COLORS[0]) : COLORS[j % COLORS.length]);
    const sumless = running || combo;  // running totals: no "all together" sums across sets of figures
    const sliceColor = (i) => (isOther(labels[i]) ? OTHER : form === "bar" ? COLORS[0] : COLORS[i % COLORS.length]);
    const total = chart.total;
    const paid = (n) => (n == null ? "" : ` · ${n} payment${n === 1 ? "" : "s"}`);

    const fig = document.createElement("figure");
    fig.className = `chart chart-${form}`;
    const cap = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = chart.title;
    const sub = document.createElement("span");
    const span = byMonth || byDay ? `${long(labels[0])} to ${long(labels[labels.length - 1])}`
      : byDom ? `days 1 to ${labels[labels.length - 1]} of each month`
      : chart.period && chart.period.from ? `${dayText(chart.period.from)} to ${dayText(chart.period.to)}` : "";
    const sofar = running ? `${series[0].name} so far` : "total";
    sub.textContent = `Money ${chart.money === "in" ? "in" : "out"} · ${sofar} ${money(total, cur)}${span ? ` · ${span}` : ""}`;
    cap.append(title, sub);
    fig.appendChild(cap);

    if (series.length > 1) {   // several sets of figures: a legend, keyed like the marks
      const legend = document.createElement("ul");
      legend.className = "chart-legend";
      series.forEach((s, j) => {
        const li = document.createElement("li");
        const key = document.createElement("span");
        const asLine = form === "line" || (combo && s.kind !== "bar");
        key.className = asLine ? "key line" : "key";
        key.style.background = combo && s.kind === "dashed"
          ? `repeating-linear-gradient(90deg, ${seriesColor(j)} 0 4px, transparent 4px 7px)` : seriesColor(j);
        if (combo && s.kind === "bar") key.style.opacity = "0.45";
        li.append(key, document.createTextNode(s.name));
        legend.appendChild(li);
      });
      fig.appendChild(legend);
    }

    const plot = document.createElement("div");
    plot.className = "chart-plot";
    const tip = document.createElement("div");
    tip.className = "chart-tip"; tip.hidden = true; tip.setAttribute("aria-hidden", "true");
    plot.appendChild(tip);
    fig.appendChild(plot);

    const foot = [...Object.entries(notes).map(([k, v]) => `${long(k)}: ${datesIn(v)}.`)];
    if (chart.note) foot.push(datesIn(chart.note));
    if (foot.length) {
      const p = document.createElement("p");
      p.className = "chart-note";
      p.textContent = `${Object.keys(notes).length ? "* " : ""}${foot.join(" ")}`;
      fig.appendChild(p);
    }

    // The table: every figure, without hovering.
    const details = document.createElement("details");
    details.className = "chart-table";
    const summary = document.createElement("summary");
    summary.textContent = "Show as a table";
    const table = document.createElement("table");
    const head = table.createTHead().insertRow();
    const word = chart.money === "in" ? "Money in" : "Money out";
    const first = byMonth ? "Month" : byDay ? "Day" : byDom ? "Day of the month"
      : chart.by === "payee" ? (chart.money === "in" ? "Payer" : "Payee") : "Category";
    const heads = series.length > 1 ? [first, ...series.map((s) => (running ? `${s.name} so far` : s.name)), ...(sumless ? [] : ["Total"])]
      : [first, word, ...(timeAxis ? [] : ["Share"]), ...(series[0].counts ? ["Payments"] : [])];
    for (const h of heads) { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); }
    const body = table.createTBody();
    labels.forEach((k, i) => {
      const row = body.insertRow();
      row.insertCell().textContent = byMonth ? `${MONTHS[monthOf(k).m]} ${monthOf(k).y}${notes[k] ? " *" : ""}`
        : byDay ? dayText(k) : k;
      if (series.length > 1) {
        series.forEach((s) => { row.insertCell().textContent = money(s.values[i], cur); });
        if (!sumless) row.insertCell().textContent = money(series.reduce((a, s) => a + s.values[i], 0), cur);
      } else {
        row.insertCell().textContent = money(series[0].values[i], cur);
        if (!timeAxis) row.insertCell().textContent = `${total ? Math.round((series[0].values[i] / total) * 100) : 0}%`;
        if (series[0].counts) row.insertCell().textContent = String(series[0].counts[i] ?? "");
      }
    });
    const foot2 = table.createTFoot().insertRow();
    foot2.insertCell().textContent = sumless ? "So far" : "Total";
    if (sumless) {   // columns add up; running totals are their last point
      series.forEach((s) => {
        const v = combo && s.kind === "bar" ? s.values.reduce((a, x) => a + x, 0) : s.values[s.values.length - 1];
        foot2.insertCell().textContent = money(v, cur);
      });
    } else {
      if (series.length > 1) series.forEach((s) => { foot2.insertCell().textContent = money(s.values.reduce((a, v) => a + v, 0), cur); });
      foot2.insertCell().textContent = money(total, cur);
    }
    while (foot2.cells.length < heads.length) foot2.insertCell();
    details.append(summary, table);
    fig.appendChild(details);

    // Hover and keyboard focus: one tip, every set of figures at that point.
    function show(rows, x, el, y) {
      tip.replaceChildren();
      for (const r of rows) {
        const line = document.createElement("div");
        if (r.color) { const key = document.createElement("span"); key.className = "key"; key.style.background = r.color; line.appendChild(key); }
        const v = document.createElement("strong"); v.textContent = r.value;
        const l = document.createElement("span"); l.textContent = r.label;
        line.append(v, l);
        tip.appendChild(line);
      }
      tip.hidden = false;
      if (timeAxis) plot.classList.add("tipping");   // the tip covers the top: hide the peak's label under it
      tip.style.top = y == null ? "" : `${Math.max(0, y - tip.offsetHeight - 4)}px`;
      tip.style.left = `${Math.max(0, Math.min(plot.clientWidth - tip.offsetWidth, x - tip.offsetWidth / 2))}px`;
      plot.querySelectorAll(".on").forEach((n) => n.classList.remove("on"));
      el.classList.add("on");
    }
    function hide() { tip.hidden = true; plot.classList.remove("tipping"); plot.querySelectorAll(".on").forEach((n) => n.classList.remove("on")); }
    function hover(el, rows, x, y) {
      el.addEventListener("pointerenter", () => show(rows, x, el, y));
      el.addEventListener("focus", () => show(rows, x, el, y));
      el.addEventListener("pointerleave", hide);
      el.addEventListener("blur", hide);
    }
    const monthRows = (i) => {
      const rows = series.map((s, j) => ({ color: series.length > 1 ? seriesColor(j) : null, value: money(s.values[i], cur),
        label: series.length > 1 ? s.name : `${long(labels[i])}${paid(s.counts && s.counts[i])}` }));
      if (combo) {
        return [{ value: "", label: long(labels[i]) }, ...series.map((s, j) => ({ color: seriesColor(j), value: money(s.values[i], cur),
          label: s.kind === "bar" ? `that day${paid(s.counts && s.counts[i])}` : s.name }))];
      }
      if (running) rows.unshift({ value: "", label: `${long(labels[i])}: spent so far` });
      else if (series.length > 1) rows.unshift({ value: money(series.reduce((a, s) => a + s.values[i], 0), cur), label: `${long(labels[i])}, all together` });
      if (notes[labels[i]]) rows.push({ value: "", label: datesIn(notes[labels[i]]) });
      return rows;
    };

    // Axes shared by the month forms: round-number gridlines and month labels that never touch.
    function frame(svg, W, H, maxV, top, axisH) {
      const ticks = niceTicks(maxV);
      const ymax = ticks[ticks.length - 1];
      const tickLabels = ticks.map((v) => money(v, cur, true));
      const left = Math.min(64, 12 + Math.max(...tickLabels.map((s) => textWidth(s, plot))));
      const y = (v) => top + (H - top) * (1 - v / ymax);
      ticks.forEach((v, i) => {
        svg.appendChild(svgEl("line", { x1: left, x2: W, y1: y(v), y2: y(v), class: i ? "grid" : "base" }));
        const t = svgEl("text", { x: left - 8, y: y(v) + 4, class: "tick", "text-anchor": "end" });
        t.textContent = tickLabels[i];
        svg.appendChild(t);
      });
      return { left, y, ymax };
    }
    function monthLabels(g, i, cx, band, H) {
      const widest = 10 + Math.max(...labels.map((k) => textWidth(`${short(k)}${flag(k)}`, plot)));
      const every = Math.max(1, Math.ceil(widest / band));
      if ((labels.length - 1 - i) % every !== 0) return;
      const t = svgEl("text", { x: cx, y: H + 18, class: "month", "text-anchor": "middle" });
      t.textContent = `${short(labels[i])}${flag(labels[i])}`;
      g.appendChild(t);
    }

    function drawColumns(svg, W) {
      const H = 190, axisH = 26, top = 22;
      const stacked = form === "stacked";
      const sums = labels.map((_, i) => series.reduce((a, s) => a + s.values[i], 0));
      const maxV = stacked ? Math.max(...sums) : Math.max(...series.flatMap((s) => s.values));
      const { left, y, ymax } = frame(svg, W, H, maxV, top, axisH);
      const band = (W - left) / labels.length;
      const n = series.length;
      const grouped = !stacked && n > 1;
      // many days in a narrow chart: keep a gap between columns rather than a 6px minimum
      const barW = grouped ? Math.max(3, Math.min(16, (band * 0.78 - 2 * (n - 1)) / n))
        : Math.max(Math.min(6, band * 0.8), Math.min(24, band * 0.6));
      const peak = (stacked ? sums : series[0].values).reduce((b, v, i, a) => (v > a[b] ? i : b), 0);
      labels.forEach((k, i) => {
        const cx = left + band * (i + 0.5);
        const g = svgEl("g", { class: "col", tabindex: "0", role: "img",
          "aria-label": monthRows(i).map((r) => `${r.label}${r.value ? `: ${r.value}` : ""}`).join("; ") });
        g.appendChild(svgEl("rect", { x: left + band * i, y: top - 18, width: band, height: H - top + 18 + axisH, class: "hit" }));
        const scale = (v) => (H - top) * (v / ymax);
        if (stacked) {
          let base = 0;
          const shown = series.map((s) => s.values[i]).map((v) => scale(v));
          const last = shown.reduce((b, h, j) => (h > 0.5 ? j : b), -1);
          series.forEach((s, j) => {
            const h = shown[j];
            if (h > 0.5) {
              const yb = y(0) - base, yt = yb - h;
              const gap = j < last ? Math.min(2, h / 2) : 0;   // a 2px sliver of background between segments
              g.appendChild(svgEl("path", { class: "seg", fill: seriesColor(j),
                d: colPath(cx - barW / 2, cx + barW / 2, yb, yt + gap, j === last) }));
            }
            base += h;
          });
        } else {
          series.forEach((s, j) => {
            const h = scale(s.values[i]);
            if (h <= 0.5) return;
            const x0 = grouped ? cx - (n * barW + 2 * (n - 1)) / 2 + j * (barW + 2) : cx - barW / 2;
            g.appendChild(svgEl("path", { class: "seg", fill: seriesColor(j), d: colPath(x0, x0 + barW, y(0), y(0) - h) }));
          });
        }
        monthLabels(g, i, cx, band, H);
        if (i === peak && !grouped && (stacked ? sums[i] : series[0].values[i]) > 0) {   // label the highest only
          const v = stacked ? sums[i] : series[0].values[i];
          const t = svgEl("text", { x: cx, y: y(v) - 6, class: "value", "text-anchor": "middle" });
          t.textContent = money(v, cur);
          g.appendChild(t);
        }
        hover(g, monthRows(i), cx);
        svg.appendChild(g);
      });
      return H + axisH;
    }

    function drawLine(svg, W) {
      const H = 190, axisH = 26, top = 22;
      const n = series.length;
      const maxV = Math.max(...series.flatMap((s) => s.values));
      // Names at the line ends only when they don't crowd each other; the legend is there either way.
      const ticks = niceTicks(maxV), top2 = ticks[ticks.length - 1];
      const endY = series.map((s) => top + (H - top) * (1 - s.values[s.values.length - 1] / top2)).sort((a, b) => a - b);
      const roomy = n > 1 && endY.every((v, k) => k === 0 || v - endY[k - 1] >= 14);
      const pad = roomy ? Math.min(120, 14 + Math.max(...series.map((s) => textWidth(s.name, plot)))) : 12;
      const { left, y } = frame(svg, W - pad, H, maxV, top, axisH);
      const band = (W - pad - left) / labels.length;
      const xs = labels.map((_, i) => left + band * (i + 0.5));
      const hair = svgEl("line", { x1: 0, x2: 0, y1: top - 8, y2: H, class: "hair" });
      svg.appendChild(hair);
      series.forEach((s, j) => {
        svg.appendChild(svgEl("path", { class: "line", stroke: seriesColor(j),
          d: s.values.map((v, i) => `${i ? "L" : "M"}${xs[i]},${y(v)}`).join("") }));
      });
      series.forEach((s, j) => {
        const last = s.values.length - 1;
        svg.appendChild(svgEl("circle", { cx: xs[last], cy: y(s.values[last]), r: 4, fill: seriesColor(j), class: "dot" }));
        if (roomy) {
          const t = svgEl("text", { x: xs[last] + 8, y: y(s.values[last]) + 4, class: "end" });
          t.textContent = s.name;
          svg.appendChild(t);
        }
      });
      if (n === 1) {
        const v = series[0].values;
        const peak = v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
        if (v[peak] > 0) {
          const t = svgEl("text", { x: xs[peak], y: y(v[peak]) - 10, class: "value", "text-anchor": "middle" });
          t.textContent = money(v[peak], cur);
          svg.appendChild(t);
        }
      }
      labels.forEach((k, i) => {
        const g = svgEl("g", { class: "col", tabindex: "0", role: "img",
          "aria-label": monthRows(i).map((r) => `${r.label}${r.value ? `: ${r.value}` : ""}`).join("; ") });
        g.appendChild(svgEl("rect", { x: xs[i] - band / 2, y: top - 18, width: band, height: H - top + 18 + axisH, class: "hit" }));
        monthLabels(g, i, xs[i], band, H);
        hover(g, monthRows(i), xs[i]);
        g.addEventListener("pointerenter", () => { hair.setAttribute("x1", xs[i]); hair.setAttribute("x2", xs[i]); });
        g.addEventListener("focus", () => { hair.setAttribute("x1", xs[i]); hair.setAttribute("x2", xs[i]); });
        svg.appendChild(g);
      });
      return H + axisH;
    }

    // Columns for each day (left scale) with running-total lines on top (right scale). The right
    // scale's ticks sit on the same gridlines as the left's, so the chart has one set of lines.
    function drawCombo(svg, W) {
      const H = 190, axisH = 26, top = 22;
      const bars = series[0].values, lines = series.slice(1);
      const barMax = Math.max(0, ...bars);
      const lineMax = Math.max(0, ...lines.flatMap((s) => s.values));
      const gridCount = niceTicks(barMax).length - 1;
      const mag = 10 ** Math.floor(Math.log10(lineMax / gridCount || 1));
      const rStep = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s * gridCount >= lineMax) || mag * 10;
      const rTop = rStep * gridCount;
      const rLabels = Array.from({ length: gridCount + 1 }, (_, i) => money(rStep * i, cur, true));
      const pad = 12 + Math.max(...rLabels.map((s) => textWidth(s, plot)));
      const { left, y } = frame(svg, W - pad, H, barMax, top, axisH);
      const yR = (v) => top + (H - top) * (1 - v / rTop);
      rLabels.forEach((s, i) => {
        const t = svgEl("text", { x: W - pad + 8, y: yR(rStep * i) + 4, class: "tick", "text-anchor": "start" });
        t.textContent = s;
        svg.appendChild(t);
      });
      const band = (W - pad - left) / labels.length;
      const xs = labels.map((_, i) => left + band * (i + 0.5));
      const barW = Math.max(Math.min(6, band * 0.8), Math.min(24, band * 0.6));
      const hair = svgEl("line", { x1: 0, x2: 0, y1: top - 8, y2: H, class: "hair" });
      svg.appendChild(hair);
      bars.forEach((v, i) => {   // the columns first, so the lines are drawn over them
        const yt = y(v);
        if (y(0) - yt > 0.5) {
          svg.appendChild(svgEl("path", { class: "seg", fill: seriesColor(0), "fill-opacity": "0.45",
            d: colPath(xs[i] - barW / 2, xs[i] + barW / 2, y(0), yt) }));
        }
      });
      lines.forEach((s, j) => {
        const attrs = { class: "line", stroke: seriesColor(j + 1), d: s.values.map((v, i) => `${i ? "L" : "M"}${xs[i]},${yR(v)}`).join("") };
        if (s.kind === "dashed") attrs["stroke-dasharray"] = "5 4";
        svg.appendChild(svgEl("path", attrs));
        const last = s.values.length - 1;
        svg.appendChild(svgEl("circle", { cx: xs[last], cy: yR(s.values[last]), r: 4, fill: seriesColor(j + 1), class: "dot" }));
      });
      labels.forEach((k, i) => {
        const rows = monthRows(i);
        const g = svgEl("g", { class: "col", tabindex: "0", role: "img",
          "aria-label": rows.map((r) => `${r.label}${r.value ? `: ${r.value}` : ""}`).join("; ") });
        g.appendChild(svgEl("rect", { x: xs[i] - band / 2, y: top - 18, width: band, height: H - top + 18 + axisH, class: "hit" }));
        monthLabels(g, i, xs[i], band, H);
        hover(g, rows, xs[i]);
        g.addEventListener("pointerenter", () => { hair.setAttribute("x1", xs[i]); hair.setAttribute("x2", xs[i]); });
        g.addEventListener("focus", () => { hair.setAttribute("x1", xs[i]); hair.setAttribute("x2", xs[i]); });
        svg.appendChild(g);
      });
      return H + axisH;
    }

    function drawBars(svg, W) {
      const values = series[0].values;
      const counts = series[0].counts || [];
      const rowH = 30, barH = 18;
      const valueW = 10 + Math.max(...values.map((v) => textWidth(money(v, cur), plot)));
      const nameW = Math.min(W * 0.42, 16 + Math.max(...labels.map((k) => textWidth(k, plot))));
      const plotW = Math.max(40, W - nameW - valueW);
      const maxV = Math.max(...values);
      svg.appendChild(svgEl("line", { x1: nameW, x2: nameW, y1: 0, y2: rowH * labels.length, class: "base" }));
      labels.forEach((k, i) => {
        const yMid = rowH * i + rowH / 2;
        const len = maxV ? (values[i] / maxV) * plotW : 0;
        const share = total ? Math.round((values[i] / total) * 100) : 0;
        const g = svgEl("g", { class: "row", tabindex: "0", role: "img",
          "aria-label": `${k}: ${money(values[i], cur)}, ${share}%${paid(counts[i])}` });
        g.appendChild(svgEl("rect", { x: 0, y: rowH * i, width: W, height: rowH, class: "hit" }));
        const name = svgEl("text", { x: nameW - 8, y: yMid + 4, class: "name", "text-anchor": "end" });
        name.textContent = fitText(k, nameW - 10, plot);
        g.appendChild(name);
        if (len > 0.5) {   // rounded at the value end, square at the baseline
          const r = Math.min(4, len, barH / 2), x0 = nameW, x1 = nameW + len, y0 = yMid - barH / 2, y1 = yMid + barH / 2;
          g.appendChild(svgEl("path", { class: "seg", fill: sliceColor(i),
            d: `M${x0},${y0}H${x1 - r}Q${x1},${y0} ${x1},${y0 + r}V${y1 - r}Q${x1},${y1} ${x1 - r},${y1}H${x0}Z` }));
        }
        const v = svgEl("text", { x: nameW + len + 6, y: yMid + 4, class: "value" });
        v.textContent = money(values[i], cur);
        g.appendChild(v);
        hover(g, [{ value: money(values[i], cur), label: `${k} · ${share}%${paid(counts[i])}` }], nameW + len / 2,
              rowH * i);   // just above its row
        svg.appendChild(g);
      });
      return rowH * labels.length + 4;
    }

    function drawDonut(svg, W) {
      const values = series[0].values;
      const counts = series[0].counts || [];
      const side = W >= 440;
      const size = Math.min(side ? 200 : 220, W);
      const R = size / 2 - 4, r0 = R * 0.62, cx = size / 2, cy = size / 2;
      let a = -Math.PI / 2;
      const pt = (rad, ang) => `${cx + rad * Math.cos(ang)},${cy + rad * Math.sin(ang)}`;
      values.forEach((v, i) => {
        const share = total ? v / total : 0;
        const a1 = a + share * 2 * Math.PI;
        const big = a1 - a > Math.PI ? 1 : 0;
        const pct = Math.round(share * 100);
        const g = svgEl("g", { class: "slice", tabindex: "0", role: "img",
          "aria-label": `${labels[i]}: ${money(v, cur)}, ${pct}%${paid(counts[i])}` });
        g.appendChild(svgEl("path", { class: "seg", fill: sliceColor(i),
          d: `M${pt(R, a)}A${R},${R} 0 ${big} 1 ${pt(R, a1)}L${pt(r0, a1)}A${r0},${r0} 0 ${big} 0 ${pt(r0, a)}Z` }));
        const mid = (a + a1) / 2;
        hover(g, [{ value: money(v, cur), label: `${labels[i]} · ${pct}%${paid(counts[i])}` }], cx + (R * 0.8) * Math.cos(mid));
        svg.appendChild(g);
        a = a1;
      });
      const t1 = svgEl("text", { x: cx, y: cy - 2, class: "centre", "text-anchor": "middle" });
      t1.textContent = money(total, cur);
      const t2 = svgEl("text", { x: cx, y: cy + 16, class: "tick", "text-anchor": "middle" });
      t2.textContent = "total";
      svg.append(t1, t2);
      svg.setAttribute("width", size); svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
      // The key, beside the ring (or under it on a phone): each slice's name, amount and share.
      plot.querySelector(".donut-key")?.remove();
      const key = document.createElement("ul");
      key.className = "donut-key";
      labels.forEach((k, i) => {
        const li = document.createElement("li");
        const sw = document.createElement("span"); sw.className = "key"; sw.style.background = sliceColor(i);
        const name = document.createElement("span"); name.className = "k"; name.textContent = k;
        const amt = document.createElement("span"); amt.className = "v";
        amt.textContent = `${money(values[i], cur)} · ${total ? Math.round((values[i] / total) * 100) : 0}%`;
        li.append(sw, name, amt);
        key.appendChild(li);
      });
      plot.classList.toggle("side", side);
      plot.appendChild(key);
      return size;
    }

    let drawnWidth = 0;
    function draw() {
      const W = Math.floor(plot.clientWidth);
      if (!W || Math.abs(W - drawnWidth) < 2) return;
      drawnWidth = W;
      const svg = svgEl("svg", { width: W, role: "group",
        "aria-label": `${chart.title}: ${form === "donut" ? "donut" : form === "bar" ? "bar" : form === "line" ? "line" : "column"} chart` });
      const height = form === "line" ? drawLine(svg, W) : form === "bar" ? drawBars(svg, W)
        : form === "donut" ? drawDonut(svg, W) : combo ? drawCombo(svg, W) : drawColumns(svg, W);
      svg.setAttribute("height", height);
      if (form !== "donut") svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
      plot.querySelector("svg")?.remove();
      plot.insertBefore(svg, tip);
    }
    if ("ResizeObserver" in window) new ResizeObserver(draw).observe(plot);
    requestAnimationFrame(draw);
    return fig;
  }

  // ---------- rendering ----------
  function scrollDown() { els.log.scrollTop = els.log.scrollHeight; }

  function addMessage(role, text, opts = {}) {
    els.empty.hidden = true;
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}${opts.error ? " error" : ""}`;
    const body = document.createElement("div");
    body.className = "text";
    if (role === "bank" && !opts.error) body.innerHTML = markdown(text);   // escaped inside markdown()
    else body.textContent = text;
    for (const chart of opts.charts || []) {
      if (chart && (chart.labels || []).length >= 2 && (chart.series || []).length) body.appendChild(renderChart(chart));
    }
    wrap.appendChild(body);
    els.log.appendChild(wrap);
    scrollDown();
    return wrap;
  }

  const stepText = (s) => (s.kind === "specialist" ? s.label : `Using ${String(s.name).replace(/_/g, " ")}`);

  function thinking() {
    els.empty.hidden = true;
    const el = document.createElement("div");
    el.className = "thinking msg";
    const label = document.createElement("p");
    label.className = "thinking-label";
    label.textContent = "Working on it";
    const steps = document.createElement("ul");
    steps.className = "progress";
    el.append(label, steps);
    els.log.appendChild(el);
    scrollDown();
    return el;
  }

  function showProgress(el, steps) {
    const list = el.querySelector(".progress");
    if (!steps || list.childElementCount === steps.length) return;
    list.replaceChildren(...steps.map((s) => {
      const li = document.createElement("li");
      li.className = s.kind;
      li.textContent = stepText(s);
      return li;
    }));
    scrollDown();
  }

  const argText = (args) => Object.entries(args || {})
    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ");

  function addTrace(question, data) {
    const li = document.createElement("li");
    li.className = "trace-item";
    const q = document.createElement("p");
    q.className = "trace-q";
    q.textContent = question;
    li.appendChild(q);
    const steps = document.createElement("ul");
    steps.className = "trace-steps";
    for (const s of data.steps || []) {
      const item = document.createElement("li");
      if (s.kind === "specialist") { item.className = "specialist"; item.textContent = s.label; }
      else {
        item.textContent = s.name;
        const a = argText(s.args);
        if (a) { const span = document.createElement("span"); span.className = "args"; span.textContent = ` (${a})`; item.appendChild(span); }
      }
      steps.appendChild(item);
    }
    if (!(data.steps || []).length) {
      const item = document.createElement("li");
      item.textContent = "Answered directly";
      steps.appendChild(item);
    }
    li.appendChild(steps);
    for (const g of data.guard || []) {
      const p = document.createElement("p");
      p.className = `trace-guard${/blocked/.test(g) ? " blocked" : ""}`;
      p.textContent = `Guardrail: ${g}`;
      li.appendChild(p);
    }
    if (data.cost) {
      const p = document.createElement("p");
      p.className = "trace-cost";
      p.textContent = data.cost;
      li.appendChild(p);
    }
    els.trace.prepend(li);
  }

  function renderSlip(slipData, code) {
    const node = els.slipTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".slip-title").textContent = slipData.title;
    node.querySelector(".slip").setAttribute("aria-label", slipData.title);
    const dl = node.querySelector(".slip-rows");
    for (const [k, v] of slipData.rows || []) {
      const row = document.createElement("div");
      const dt = document.createElement("dt"); dt.textContent = k;
      const leader = document.createElement("span"); leader.className = "leader"; leader.setAttribute("aria-hidden", "true");
      const dd = document.createElement("dd"); dd.textContent = v;
      row.append(dt, leader, dd);
      dl.appendChild(row);
    }
    const approve = node.querySelector(".approve");
    const decline = node.querySelector(".decline");
    const codeForm = node.querySelector(".slip-code");
    const codeInput = node.querySelector(".code-input");
    const codeNote = node.querySelector(".code-note");
    approve.textContent = slipData.approve_label || "Approve";
    els.empty.hidden = true;
    els.log.appendChild(node);
    scrollDown();

    const showCodeBox = (sentTo, expiresIn) => {
      codeForm.hidden = false;
      const minutes = Math.max(1, Math.round(expiresIn / 60));
      node.querySelector(".code-label").textContent =
        `Enter the 6-digit code we emailed to ${sentTo}. It expires in ${minutes} minute${minutes === 1 ? "" : "s"}.`;
      node.querySelector(".code-label").setAttribute("for", codeInput.id = `code-${Date.now()}`);
      approve.textContent = "Confirm code";
      codeInput.focus();
    };
    if (code) showCodeBox(code.sent_to, code.expires_in); else approve.focus();

    const lock = (on) => { approve.disabled = decline.disabled = on; codeInput.disabled = on; };
    const close = (outcome) => {
      node.classList.add("done");
      const p = document.createElement("p");
      p.className = "slip-outcome";
      p.textContent = outcome;
      node.querySelector(".slip").appendChild(p);
    };

    async function decide(approved) {
      if (busy) return;
      const typed = codeForm.hidden ? undefined : codeInput.value.trim();
      if (approved && !codeForm.hidden && !/^\d{6}$/.test(typed)) { codeNote.textContent = "Enter the 6-digit code from the email."; codeInput.focus(); return; }
      busy = true; lock(true); setComposer(false);
      let wait = null;
      try {
        const data = await api("/api/confirm", { thread_id: threadId, approve: approved, code: typed });
        if (data.status === "code_required") { lock(false); showCodeBox(data.sent_to, data.expires_in); return; }
        if (data.status === "wrong_code") {
          codeNote.textContent = `That code isn't right. ${data.attempts_left} ${data.attempts_left === 1 ? "try" : "tries"} left.`;
          codeInput.value = ""; lock(false); codeInput.focus();
          return;
        }
        close(approved ? "Approved." : "Declined.");
        if (data.status === "needs_confirmation") { renderSlip(data.slip); return; }   // another action in the same step
        wait = thinking();
        handle(slipData.title, await follow(data.turn_id, wait));
        refreshAllowance();
      } catch (err) {
        if (node.classList.contains("done")) addMessage("bank", err.message, { error: true });
        else { lock(false); codeNote.textContent = err.message; }
      } finally {
        if (wait) wait.remove();
        busy = false; setComposer(true);
      }
    }
    approve.addEventListener("click", () => decide(true));
    decline.addEventListener("click", () => decide(false));
    codeForm.addEventListener("submit", (e) => { e.preventDefault(); decide(true); });
  }

  function handle(question, data) {
    if (data.thread_id) { threadId = data.thread_id; store.set(threadKey(), threadId); }
    if (data.steps || data.cost) addTrace(question, data);
    if (data.status === "needs_confirmation") renderSlip(data.slip);
    else if (data.status === "answered") addMessage("bank", data.answer, { charts: data.charts });
    else if (data.status === "failed") addMessage("bank", data.error || "I couldn't finish that. Please try again.", { error: true });
    refreshConversations();
  }

  // ---------- your conversations: newest first; open one to carry on where you left off ----------
  let conversations = [];
  let showAllConvos = false;
  const CONVOS_SHOWN = 8;

  function whenText(ts) {
    const d = new Date(ts * 1000), now = new Date();
    const hm = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const days = Math.round((day(now) - day(d)) / 86400000);
    if (days === 0) return `Today ${hm}`;
    if (days === 1) return `Yesterday ${hm}`;
    return `${d.getDate()} ${MONTHS[d.getMonth()]}${d.getFullYear() !== now.getFullYear() ? ` ${d.getFullYear()}` : ""}`;
  }

  function renderConversations() {
    els.convos.replaceChildren();
    if (!conversations.length) {
      const li = document.createElement("li");
      li.className = "convo-none";
      li.textContent = "Your conversations will appear here.";
      els.convos.appendChild(li);
      return;
    }
    for (const c of showAllConvos ? conversations : conversations.slice(0, CONVOS_SHOWN)) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button"; b.className = "convo";
      if (c.thread_id === threadId) b.setAttribute("aria-current", "true");
      const t = document.createElement("span"); t.className = "convo-title"; t.textContent = c.title;
      const m = document.createElement("span"); m.className = `convo-meta${c.waiting ? " waiting" : ""}`;
      m.textContent = whenText(c.updated) + (c.waiting ? " · waiting for you" : c.running ? " · answering" : "");
      b.append(t, m);
      b.title = c.title;
      b.addEventListener("click", () => openConversation(c.thread_id));
      li.appendChild(b);
      els.convos.appendChild(li);
    }
    if (conversations.length > CONVOS_SHOWN) {
      const li = document.createElement("li");
      const more = document.createElement("button");
      more.type = "button"; more.className = "text-button convo-more";
      more.textContent = showAllConvos ? "Show fewer" : `Show all ${conversations.length}`;
      more.addEventListener("click", () => { showAllConvos = !showAllConvos; renderConversations(); });
      li.appendChild(more);
      els.convos.appendChild(li);
    }
  }

  async function refreshConversations() {
    if (!me || me.choosing) return;
    try { conversations = (await api("/api/conversations")).conversations || []; } catch { return; }
    renderConversations();
  }

  async function openConversation(id) {
    if (busy) return;
    closeMenu(false);
    if (id === threadId) { els.input.focus(); return; }
    resetConversation(false);
    store.set(threadKey(), id);
    await restore();
    renderConversations();
    if (window.matchMedia("(max-width: 52rem)").matches) window.scrollTo({ top: 0 });
    els.input.focus();
  }

  function newConversation() {
    if (busy) return;
    closeMenu(false);
    resetConversation();
    renderConversations();
    els.input.focus();
  }

  // On narrower screens the list is a panel that slides in from the left (the menu button).
  const narrow = window.matchMedia("(max-width: 64rem)");
  function openMenu() {
    if (!narrow.matches) return;
    els.panel.classList.add("open");
    els.scrim.hidden = false;
    els.menu.setAttribute("aria-expanded", "true");
    for (const el of [els.conversation, els.traceAside]) el.inert = true;   // the rest waits behind the panel
    (els.panel.querySelector('.convo[aria-current="true"]') || els.convoClose).focus();
  }
  function closeMenu(focusButton = true) {
    if (!els.panel.classList.contains("open")) return;
    els.panel.classList.remove("open");
    els.scrim.hidden = true;
    els.menu.setAttribute("aria-expanded", "false");
    for (const el of [els.conversation, els.traceAside]) el.inert = false;
    if (focusButton) els.menu.focus();
  }

  function setComposer(on) { els.send.disabled = !on; }

  // ---------- visitors: their pretend bank, its time left and today's messages ----------
  function timeLeft(expires) {
    const minutes = Math.max(0, Math.round((expires * 1000 - Date.now()) / 60000));
    if (minutes >= 90) return `in about ${Math.round(minutes / 60)} hours`;
    return `in ${minutes} minute${minutes === 1 ? "" : "s"}`;
  }

  function renderNotice() {
    if (!me || me.kind !== "guest") { els.notice.hidden = true; return; }
    els.notice.hidden = false;
    els.notice.textContent = `This is your own pretend bank: no real money, and it's deleted ${timeLeft(me.expires)}. `
      + `${me.messages_left} of ${config.visitors.daily_messages} messages left today.`;
  }

  async function refreshAllowance() {
    if (!me || me.kind !== "guest") return;
    try { me = { ...me, ...(await api("/api/me")) }; renderNotice(); } catch { /* shown elsewhere */ }
  }

  // ---------- actions ----------
  async function send(text) {
    text = text.trim();
    if (!text || busy) return;
    busy = true; setComposer(false);
    addMessage("you", text);
    els.input.value = ""; autosize();
    const wait = thinking();
    try {
      const started = await api("/api/chat", { message: text, thread_id: threadId });
      if (started.thread_id) { threadId = started.thread_id; store.set(threadKey(), threadId); }
      refreshConversations();   // a new conversation appears in the list straight away
      if (started.httpStatus === 409 && started.status === "needs_confirmation") {
        addMessage("bank", started.detail, { error: true });
      } else {
        // 409 "running": an earlier message is still being answered, so show that answer.
        handle(text, await follow(started.turn_id, wait));
      }
      refreshAllowance();
    } catch (err) {
      addMessage("bank", err.message, { error: true });
    } finally {
      wait.remove();
      busy = false; setComposer(true); els.input.focus();
    }
  }

  function resetConversation(forget = true) {
    threadId = null;
    if (forget) store.del(threadKey());
    els.log.querySelectorAll(".msg, .slip-wrap, .thinking").forEach((n) => n.remove());
    els.trace.replaceChildren();
    els.empty.hidden = false;
  }

  async function restore() {
    const saved = store.get(threadKey());
    if (!saved) return;
    let data;
    try {
      data = await api(`/api/history?thread_id=${encodeURIComponent(saved)}`);
    } catch { store.del(threadKey()); return; }
    threadId = data.thread_id;
    let lastQuestion = "";
    for (const m of data.messages) {
      addMessage(m.role, m.text, { charts: m.charts });
      if (m.role === "you") lastQuestion = m.text;
    }
    if (data.slip) renderSlip(data.slip, data.code);
    if (data.running_turn) {   // reloaded while an answer was on its way
      busy = true; setComposer(false);
      const wait = thinking();
      try { handle(lastQuestion, await follow(data.running_turn, wait)); }
      catch (err) { addMessage("bank", err.message, { error: true }); }
      finally { wait.remove(); busy = false; setComposer(true); }
    }
  }

  async function loadConfig() {
    try { config = await (await fetch("/api/config", { credentials: "same-origin" })).json(); } catch { /* offline */ }
  }

  const isPublic = () => config.auth_mode === "public";
  const signinNote = (text) => { (isPublic() ? els.publicNote : els.visitorNote).textContent = text || ""; };

  async function showSignIn(message) {
    me = null;
    els.chat.hidden = true; els.who.hidden = true; els.choose.hidden = true; els.signIn.hidden = false;
    closeMenu(false); els.menu.hidden = true;
    await loadConfig();
    els.publicSignin.hidden = !isPublic();
    els.visitorForm.hidden = isPublic();
    const google = isPublic() && config.signin && config.signin.google;
    els.googleButton.hidden = !google;
    els.orDivider.hidden = !google;
    showEmailStep();
    signinNote(message && message !== "Please sign in." ? message : "");
    const v = config.visitors || { bank_hours: 24, daily_messages: 30 };
    els.visitorTerms.textContent = `Your demo bank is deleted ${v.bank_hours} hours after you open it, with your `
      + `conversations. Up to ${v.daily_messages} messages a day. It's all pretend: please don't type real card `
      + "numbers or passwords.";
    els.customers.replaceChildren();
    els.owners.hidden = !(config.customers || []).length;   // the development sign-in only
    for (const c of config.customers || []) {
      els.customers.appendChild(customerButton(c, async () => {
        await api("/api/login", { customer: c.customer });
        await enter();
      }, (err) => signinNote(err.message)));
    }
    (isPublic() ? (google ? els.googleButton : els.signinEmail) : els.visitorEmail).focus();
  }

  function customerButton(c, choose, fail) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "customer";
    const name = document.createElement("strong"); name.textContent = c.name;
    const label = document.createElement("span"); label.textContent = `Demo customer ${c.customer}`;
    b.append(name, label);
    b.addEventListener("click", async () => {
      b.disabled = true;
      try { await choose(); } catch (err) { b.disabled = false; fail(err); }
    });
    return b;
  }

  // The owner (an address in OWNER_EMAILS) picks Robert or Anil, and can switch later.
  function showChooser() {
    els.signIn.hidden = true; els.chat.hidden = true; els.choose.hidden = false;
    closeMenu(false); els.menu.hidden = true;
    els.who.hidden = me.choosing; els.chooseNote.textContent = "";
    els.chooseTitle.textContent = me.choosing ? `Welcome back, ${me.name}` : "Switch customer";
    els.chooseCustomers.replaceChildren(...(me.customers || []).map((c) => customerButton(c, async () => {
      await api("/api/owner/customer", { customer: c.customer });
      await enter();
    }, (err) => { els.chooseNote.textContent = err.message; })));
    const first = els.chooseCustomers.querySelector("button");
    if (first) first.focus();
  }

  function showEmailStep() {
    signinChallenge = null;
    els.emailForm.hidden = false; els.codeForm.hidden = true; els.signinCode.value = "";
  }

  async function enter() {
    me = await api("/api/me");
    if (me.choosing) showChooser(); else await showChat();
  }

  async function showChat() {
    els.signIn.hidden = true; els.choose.hidden = true; els.chat.hidden = false; els.who.hidden = false;
    els.menu.hidden = false;
    const customer = (me.customers || []).find((c) => c.customer === me.customer);
    els.whoName.textContent = customer && customer.name !== me.name ? `${me.name} as ${customer.name}` : me.name;
    els.resetBank.hidden = me.kind !== "guest";
    els.switchCustomer.hidden = !(me.kind === "owner" && (me.customers || []).length);
    renderNotice();
    resetConversation(false);   // clear the page, keep the saved conversation id
    conversations = []; showAllConvos = false;
    renderConversations();
    await restore();
    await refreshConversations();
    if (window.matchMedia("(max-width: 52rem)").matches) $("trace-toggle").open = false;
    els.input.focus();
  }

  function autosize() { els.input.style.height = "auto"; els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`; }

  // ---------- wiring ----------
  els.form.addEventListener("submit", (e) => { e.preventDefault(); send(els.input.value); });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(els.input.value); }
  });
  els.input.addEventListener("input", autosize);
  document.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
  els.convoNew.addEventListener("click", newConversation);
  els.menu.addEventListener("click", () => (els.panel.classList.contains("open") ? closeMenu() : openMenu()));
  els.convoClose.addEventListener("click", () => closeMenu());
  els.scrim.addEventListener("click", () => closeMenu());
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
  narrow.addEventListener("change", () => closeMenu(false));   // e.g. a tablet turned sideways
  els.signOut.addEventListener("click", async () => {
    try { await api("/api/logout", {}); } catch { /* already signed out */ }
    store.del(threadKey());
    showSignIn();
  });
  els.resetBank.addEventListener("click", async () => {
    if (busy || !window.confirm("Start over with a new pretend bank? Your balances and conversations will be replaced.")) return;
    busy = true; setComposer(false);
    try {
      await api("/api/guest/reset", {});
      resetConversation();
      await refreshAllowance();
      await refreshConversations();
    } catch (err) {
      addMessage("bank", err.message, { error: true });
    } finally { busy = false; setComposer(true); }
  });
  els.switchCustomer.addEventListener("click", () => { if (!busy) showChooser(); });
  els.emailForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = els.signinEmail.value.trim();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) { signinNote("Enter your email address."); els.signinEmail.focus(); return; }
    els.emailGo.disabled = true; signinNote("");
    try {
      const sent = await api("/api/auth/email/start", { email });
      signinChallenge = sent.challenge;
      els.emailForm.hidden = true; els.codeForm.hidden = false;
      els.signinCodeLabel.textContent = `Enter the 6-digit code we emailed to ${sent.sent_to}. `
        + `It expires in ${Math.round(sent.expires_in / 60)} minutes.`;
      els.signinCode.focus();
    } catch (err) {
      signinNote(err.message);
    } finally { els.emailGo.disabled = false; }
  });
  els.codeForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const code = els.signinCode.value.trim();
    if (!/^\d{6}$/.test(code)) { signinNote("Enter the 6-digit code from the email."); els.signinCode.focus(); return; }
    els.codeGo.disabled = true; signinNote("");
    try {
      await api("/api/auth/email/check", { challenge: signinChallenge, code });
      await enter();
    } catch (err) {
      const d = err.data || {};
      if (d.error === "wrong_code") {
        signinNote(`That code isn't right. ${d.attempts_left} ${d.attempts_left === 1 ? "try" : "tries"} left.`);
        els.signinCode.value = ""; els.signinCode.focus();
      } else if (d.error) {
        showEmailStep();
        signinNote("That code has expired or had too many tries. Ask for a new one.");
      } else signinNote(err.message);
    } finally { els.codeGo.disabled = false; }
  });
  els.codeRestart.addEventListener("click", () => { showEmailStep(); signinNote(""); els.signinEmail.focus(); });
  els.visitorForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = els.visitorEmail.value.trim();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
      els.visitorNote.textContent = "Enter your email address.";
      els.visitorEmail.focus();
      return;
    }
    els.visitorGo.disabled = true; els.visitorNote.textContent = "";
    try {
      await api("/api/login", { email, name: els.visitorName.value.trim() || undefined });
      await enter();
    } catch (err) {
      els.visitorNote.textContent = err.message;
    } finally { els.visitorGo.disabled = false; }
  });

  (async () => {
    await loadConfig();
    const error = new URLSearchParams(window.location.search).get("signin_error");
    if (error !== null) window.history.replaceState(null, "", window.location.pathname);   // back from Google
    try {
      const res = await fetch("/api/me", { credentials: "same-origin" });
      if (res.ok) { me = await res.json(); if (me.choosing) showChooser(); else showChat(); }
      else showSignIn(error !== null ? (SIGNIN_ERRORS[error] || SIGNIN_ERRORS.google) : undefined);
    } catch { showSignIn(); }
  })();
})();
