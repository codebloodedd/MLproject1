function basePath() {
  // If your site is served from /<repo>/ and your files are in /website,
  // then index.html URL will include "/website/".
  const path = window.location.pathname;
  return path.includes("/website/") ? "website/" : "";
}

async function loadPartial(id, relPath) {
  const el = document.getElementById(id);
  if (!el) return;

  const base = basePath();
  const url = base + relPath;

  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    el.innerHTML = await res.text();
  } catch (err) {
    el.innerHTML = `<div style="padding:12px;border:1px solid rgba(255,255,255,0.15);border-radius:12px;margin:12px;max-width:1100px;color:#b8c2e6;">
      <b>Nav/Footer failed to load.</b><br/>
      Tried: <code>${url}</code><br/>
      Error: <code>${String(err)}</code><br/><br/>
      Fix: ensure <code>partials/nav.html</code> and <code>partials/footer.html</code> exist inside the same folder as this page, and open via a server (not file://).
    </div>`;
  }
}

function setActiveNav() {
  const current = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".tabs a[data-page]").forEach((a) => {
    if (a.getAttribute("data-page") === current) a.classList.add("active");
  });
}

(async function initLayout() {
  await loadPartial("nav-slot", "partials/nav.html");
  await loadPartial("footer-slot", "partials/footer.html");
  setActiveNav();
})();
