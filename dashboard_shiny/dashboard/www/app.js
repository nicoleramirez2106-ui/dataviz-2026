// Navegación, chips de filtro y contador del Inicio
(function () {
  function contar() {
    $('.contador').each(function () {
      var el = $(this);
      var v = parseFloat(el.attr('data-valor'));
      var d = parseInt(el.attr('data-dec'), 10) || 0;
      var s = el.attr('data-sufijo') || '';
      if (isNaN(v)) { return; }
      var t0 = null;
      function paso(ts) {
        if (t0 === null) { t0 = ts; }
        var k = Math.min((ts - t0) / 900, 1);
        var x = v * (1 - Math.pow(1 - k, 3));
        el.text(x.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d }) + s);
        if (k < 1) { requestAnimationFrame(paso); }
      }
      requestAnimationFrame(paso);
    });
  }

  function ir(pagina) {
    $('.riel-item').removeClass('active').filter('[data-ir="' + pagina + '"]').addClass('active');
    $('#filtros').toggleClass('oculto', !(pagina === 'dashboard' || pagina === 'analisis'));
    if (window.Shiny && Shiny.setInputValue) {
      Shiny.setInputValue('ir', pagina, { priority: 'event' });
    }
    window.scrollTo(0, 0);
    if (pagina === 'inicio') { setTimeout(contar, 80); }
  }

  $(document).on('click', '[data-ir]', function (e) {
    e.preventDefault();
    ir($(this).attr('data-ir'));
  });

  $(document).on('click', '.chips .chip', function () {
    var caja = $(this).closest('.chips');
    caja.find('.chip').removeClass('chip-activo');
    $(this).addClass('chip-activo');
    Shiny.setInputValue(caja.attr('data-chip'), $(this).attr('data-val'));
  });

  $(function () {
    if (window.Shiny && Shiny.addCustomMessageHandler) {
      Shiny.addCustomMessageHandler('chips_set', function (m) {
        var caja = $('.chips[data-chip="' + m.id + '"]');
        caja.find('.chip').removeClass('chip-activo').filter(function () {
          return $(this).attr('data-val') === m.value;
        }).addClass('chip-activo');
        Shiny.setInputValue(m.id, m.value);
      });
    }
    setTimeout(contar, 300);
  });
})();
