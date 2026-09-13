// Client-side autocomplete for the big pickers.
//
// A picker is an <input> plus a menu, holding at most ten prefix matches for
// what has been typed, from lists that live in the page — no request leaves
// the browser until "recommend". Modelled on the answers search at
// spreadthewordlist.pages.dev: matched prefix in bold, arrows and Enter,
// Escape and click-away to close, nothing shown until something is typed.
//
// Focusing an empty box shows the ten commonest entries, so the list can be
// browsed as well as searched. A label may carry a note after "  ·  " — "37
// games", "512 ratings" — shown dimmed beside the name and left off the chips;
// only the name is matched.
//
// Each picker mirrors its value into a hidden Gradio textbox (a single id, or a
// JSON array for multi-select) and fires an `input` event on it, which is how
// the value reaches Python. Python talks back through `IF.sync` — wired as a
// client-side `change` handler on the same textbox — so a reset or a mode
// change clears the widget too. The filters' lists change with every result
// set and arrive through `IF.setLists`.
window.IF = (() => {
  const LIMIT = 10;
  const lists = {};       // name -> [[label, value], ...], most common first
  const labels = {};      // name -> Map(value -> label)
  const widgets = {};     // name -> Widget
  let ready = false;

  const NOTE = "  ·  ";
  const norm = (s) => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const nameOf = (label) => { const i = label.indexOf(NOTE); return i < 0 ? label : label.slice(0, i); };
  // Just the number: "37 games" -> "37". The hint above each box says what
  // is being counted, so the word would only be noise ten times over.
  const noteOf = (label) => { const i = label.indexOf(NOTE); return i < 0 ? "" : label.slice(i + NOTE.length).replace(/[^\d].*$/, ""); };

  function search(name, key) {
    const list = lists[name] || [];
    const k = norm(key);
    const out = [];
    for (const item of list) {
      if (!k || norm(nameOf(item[0])).startsWith(k)) {
        out.push(item);
        if (out.length >= LIMIT) break;
      }
    }
    return out;
  }

  // How many characters of `shown` cover the typed key, so the bold prefix
  // survives accents and case.
  function prefixLength(shown, key) {
    const k = norm(key);
    for (let i = 1; i <= shown.length; i++) {
      if (norm(shown.slice(0, i)).length >= k.length) return i;
    }
    return shown.length;
  }

  function setList(name, pairs) {
    lists[name] = pairs;
    labels[name] = new Map(pairs.map(([l, v]) => [String(v), l]));
  }

  function box(name) {
    const host = document.getElementById(`if-box-${name}`);
    return host ? host.querySelector("textarea, input") : null;
  }

  class Widget {
    constructor(root) {
      this.root = root;
      this.name = root.dataset.name;
      this.multi = root.dataset.multi === "1";
      this.custom = root.dataset.custom === "1";
      this.input = root.querySelector("input");
      this.menu = root.querySelector(".if-ac__menu");
      this.chips = root.querySelector(".if-ac__chips");
      this.values = [];      // multi
      this.value = "";       // single
      this.picked = -1;
      this.input.disabled = !ready && !this.custom;

      this.input.addEventListener("input", () => { this.picked = -1; this.paint(); if (!this.multi) this.setSingle("", true); });
      this.input.addEventListener("keydown", (e) => this.onKey(e));
      this.input.addEventListener("focus", () => this.paint());
      this.input.addEventListener("click", () => { if (this.menu.hidden) this.paint(); });
      root.addEventListener("click", (e) => { if (e.target === root || e.target.classList.contains("if-ac__field")) this.input.focus(); });
      document.addEventListener("click", (e) => { if (!root.contains(e.target)) this.close(); });
      this.readBack();
    }

    // Take the current value from the hidden textbox: on first mount, and on
    // remount after a mode switch hid and re-showed the component.
    readBack() {
      const b = box(this.name);
      this.sync(b ? b.value : "");
    }

    options() { return [...this.menu.querySelectorAll(".if-ac__opt")]; }

    onKey(e) {
      const items = this.options();
      if (e.key === "Escape") { this.close(); return; }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        if (!items.length) return;
        e.preventDefault();
        const step = e.key === "ArrowDown" ? 1 : -1;
        this.picked = (this.picked + step + items.length) % items.length;
        this.highlight();
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        if (items.length) {
          const item = items[this.picked >= 0 ? this.picked : 0];
          this.choose(item.dataset.value, item.dataset.label);
        } else if (this.custom && this.input.value.trim()) {
          this.choose(this.input.value.trim(), this.input.value.trim());
        } else if (!this.multi && this.value) {
          // A chosen game and a second Enter: run the search.
          const go = document.querySelector("#action-row button.primary");
          if (go) go.click();
        }
        return;
      }
      if (e.key === "Backspace" && this.multi && this.input.value === "" && this.values.length) {
        this.remove(this.values[this.values.length - 1]);
      }
    }

    paint() {
      const key = this.input.value.trim();
      if (!ready) { this.close(); return; }
      const hits = search(this.name, key).filter(([, v]) => !this.multi || !this.values.includes(String(v)));
      if (!hits.length) { this.close(); return; }
      const frag = document.createDocumentFragment();
      hits.forEach(([label, value], i) => {
        const row = document.createElement("div");
        row.className = "if-ac__opt";
        row.id = `if-opt-${this.name}-${i}`;
        row.setAttribute("role", "option");
        row.dataset.value = value;
        row.dataset.label = label;
        const name = nameOf(label), note = noteOf(label);
        const text = document.createElement("span");
        text.className = "if-ac__opt-name";
        const cut = key ? prefixLength(name, key) : 0;
        if (cut) {
          const head = document.createElement("b");
          head.textContent = name.slice(0, cut);
          text.appendChild(head);
        }
        text.appendChild(document.createTextNode(name.slice(cut)));
        row.appendChild(text);
        if (note) {
          const aside = document.createElement("span");
          aside.className = "if-ac__opt-note";
          aside.textContent = note;
          row.appendChild(aside);
        }
        row.addEventListener("mousedown", (e) => { e.preventDefault(); this.choose(value, label); });
        frag.appendChild(row);
      });
      this.menu.replaceChildren(frag);
      this.menu.hidden = false;
      this.input.setAttribute("aria-expanded", "true");
      this.highlight();
    }

    highlight() {
      this.options().forEach((item, i) => {
        const on = i === this.picked;
        item.classList.toggle("is-picked", on);
        item.setAttribute("aria-selected", on ? "true" : "false");
      });
    }

    close() {
      this.menu.hidden = true;
      this.menu.replaceChildren();
      this.input.setAttribute("aria-expanded", "false");
      this.picked = -1;
    }

    choose(value, label) {
      this.close();
      if (this.multi) {
        value = String(value);
        if (!this.values.includes(value)) this.values.push(value);
        this.input.value = "";
        this.renderChips();
        this.push(JSON.stringify(this.values));
      } else {
        this.setSingle(value, false, label);
      }
    }

    setSingle(value, keepText, label) {
      this.value = String(value || "");
      if (!keepText) this.input.value = this.value ? nameOf(label ?? this.labelFor(this.value)) : "";
      this.push(this.value);
    }

    remove(value) {
      this.values = this.values.filter((v) => v !== value);
      this.renderChips();
      this.push(JSON.stringify(this.values));
      this.input.focus();
    }

    labelFor(value) {
      const m = labels[this.name];
      return (m && m.get(String(value))) || String(value);
    }

    renderChips() {
      if (!this.chips) return;
      const frag = document.createDocumentFragment();
      for (const v of this.values) {
        const chip = document.createElement("span");
        chip.className = "if-ac__chip";
        chip.textContent = nameOf(this.labelFor(v));
        const x = document.createElement("button");
        x.type = "button";
        x.className = "if-ac__chip-x";
        x.setAttribute("aria-label", `remove ${nameOf(this.labelFor(v))}`);
        x.textContent = "×";
        x.addEventListener("click", () => this.remove(v));
        chip.appendChild(x);
        frag.appendChild(chip);
      }
      this.chips.replaceChildren(frag);
      this.chips.hidden = !this.values.length;
    }

    // Widget -> Gradio.
    push(text) {
      const b = box(this.name);
      if (!b || b.value === text) return;
      b.value = text;
      b.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // Gradio -> widget: adopt a value without pushing it back.
    sync(text) {
      text = text == null ? "" : String(text);
      if (this.multi) {
        let vals = [];
        if (text.trim()) {
          try { vals = JSON.parse(text); } catch (_) { vals = [text]; }
        }
        this.values = (Array.isArray(vals) ? vals : [vals]).map(String);
        this.renderChips();
      } else {
        this.value = text;
        this.input.value = text ? nameOf(this.labelFor(text)) : "";
      }
    }
  }

  function mount(scope) {
    for (const root of (scope || document).querySelectorAll(".if-ac:not([data-mounted])")) {
      root.dataset.mounted = "1";
      widgets[root.dataset.name] = new Widget(root);
    }
  }

  function setReady() {
    ready = true;
    for (const w of Object.values(widgets)) { w.input.disabled = false; w.sync(box(w.name)?.value ?? ""); }
  }

  async function load(url) {
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      const data = await res.json();
      for (const [name, pairs] of Object.entries(data)) setList(name, pairs);
    } catch (err) {
      console.error("picker lists failed to load", err);
    }
    setReady();
  }

  // Gradio renders components after the page loads, and re-renders an HTML
  // component when it is shown again, so mount whatever appears.
  new MutationObserver(() => mount()).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", () => mount());

  return {
    load,
    mount,
    sync(name, text) { const w = widgets[name]; if (w) w.sync(text); },
    // The four filters' lists, from the current result set, as [label, value].
    setLists(named) {
      for (const [name, pairs] of Object.entries(named || {})) setList(name, pairs);
    },
  };
})();
