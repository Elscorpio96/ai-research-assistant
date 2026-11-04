// Progressive enhancement only: every action works without JavaScript.
(function () {
  'use strict';
  document.addEventListener('submit', function (e) {
    var f = e.target;
    var msg = f.getAttribute('data-confirm');
    if (msg && !window.confirm(msg)) { e.preventDefault(); return; }
    var busy = f.getAttribute('data-busy');
    if (busy) {
      var b = f.querySelector('button.primary');
      if (b) { b.disabled = true; b.textContent = busy; }
    }
  });
  document.querySelectorAll('select[data-autosubmit]').forEach(function (s) {
    s.addEventListener('change', function () { s.form.submit(); });
  });
  // Exclude forms: choosing a reason submits straight away
  document.querySelectorAll('form.exclude select').forEach(function (s) {
    s.addEventListener('change', function () { if (s.value) s.form.submit(); });
  });
})();
