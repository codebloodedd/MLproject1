const NAV_HTML = `
<div class="nav">
  <div class="nav-inner">
    <div class="tabs full">
      <a data-page="index.html" href="index.html">Introduction</a>
      <a data-page="dataprep_eda.html" href="dataprep_eda.html">Data Prep/EDA</a>
      <a data-page="pca.html" href="pca.html">PCA</a>
      <a data-page="clustering.html" href="clustering.html">Clustering</a>
      <a data-page="arm.html" href="arm.html">ARM</a>
      <a data-page="dt.html" href="dt.html">DT</a>
      <a data-page="nb.html" href="nb.html">NB</a>
      <a data-page="regression.html" href="regression.html">Regression</a>
      <a data-page="svm.html" href="svm.html">SVM</a>
      <a data-page="ensemble.html" href="ensemble.html">Ensemble</a>
      <a data-page="conclusions.html" href="conclusions.html">Conclusions</a>
      <a data-page="about.html" href="about.html">About Me</a>
    </div>
  </div>
</div>
`;

const FOOTER_HTML = `
<footer class="footer">
  <div class="footer-inner">
    <div>
      <div class="footer-title">Pratham Tushar Shah</div>
      <div class="footer-sub">Machine Learning Project - Final Website Submission (Modules 1 to 4)</div>
    </div>
    <div class="footer-links">
      <a href="https://willowy-tiramisu-66795c.netlify.app/" target="_blank" rel="noreferrer">Portfolio</a>
      <a href="https://www.linkedin.com/in/pratham-s-249611396/" target="_blank" rel="noreferrer">LinkedIn</a>
      <a href="https://github.com/codebloodedd" target="_blank" rel="noreferrer">GitHub</a>
    </div>
  </div>
</footer>
`;

function setActiveNav() {
  const current = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".tabs a[data-page]").forEach((a) => {
    if (a.getAttribute("data-page") === current) {
      a.classList.add("active");
    }
  });
}

function injectLayout() {
  const navSlot = document.getElementById("nav-slot");
  const footerSlot = document.getElementById("footer-slot");

  if (navSlot) {
    navSlot.innerHTML = NAV_HTML;
  }

  if (footerSlot) {
    footerSlot.innerHTML = FOOTER_HTML;
  }

  setActiveNav();
}

injectLayout();
