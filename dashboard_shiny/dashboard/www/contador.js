/* Números que suben contando al aparecer (marcador de la portada).
   Cada número lleva data-valor (cifra final), data-dec (decimales) y data-sufijo (" %").
   Dash carga automáticamente todo lo que está en assets/. */
(function () {
  var reducir = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var formato = function (x, dec) {
    return x.toLocaleString("de-DE", { minimumFractionDigits: dec, maximumFractionDigits: dec });
  };

  function contar(el) {
    el.setAttribute("data-hecho", "1");
    if (reducir) return;
    var final = parseFloat(el.getAttribute("data-valor"));
    var dec = parseInt(el.getAttribute("data-dec") || "0", 10);
    var sufijo = el.getAttribute("data-sufijo") || "";
    var duracion = 1600, inicio = null;
    function paso(t) {
      if (inicio === null) inicio = t;
      var p = Math.min((t - inicio) / duracion, 1);
      var suave = 1 - Math.pow(1 - p, 3);
      el.textContent = formato(final * suave, dec) + sufijo;
      if (p < 1) requestAnimationFrame(paso);
    }
    requestAnimationFrame(paso);
  }

  new MutationObserver(function () {
    document.querySelectorAll(".contador:not([data-hecho])").forEach(contar);
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
