(function () {
  'use strict';

  var root = document.documentElement;
  var storage = {
    get: function (key) {
      try {
        return window.localStorage.getItem(key);
      } catch (error) {
        return null;
      }
    },
    set: function (key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (error) {}
    }
  };

  var THEME_KEY = 'inference-textbook-theme';
  var SIZE_KEY = 'inference-textbook-textsize';
  var sizes = ['small', 'normal', 'large', 'xlarge'];
  var sizeLabels = {
    small: 'small',
    normal: 'normal',
    large: 'large',
    xlarge: 'extra large'
  };

  var themeButton = document.getElementById('theme-toggle');
  var themeLabel = document.getElementById('theme-label');
  var sizeStatus = document.getElementById('text-size-status');

  function currentTheme() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function applyTheme(theme, persist) {
    root.setAttribute('data-theme', theme);
    if (themeButton) {
      themeButton.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
      themeButton.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      themeButton.setAttribute('title', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    }
    if (themeLabel) {
      themeLabel.textContent = theme === 'dark' ? 'Light' : 'Dark';
    }
    if (persist) {
      storage.set(THEME_KEY, theme);
    }
  }

  function currentSize() {
    var value = root.getAttribute('data-textsize');
    return sizes.indexOf(value) >= 0 ? value : 'normal';
  }

  function updateSizeButtons() {
    var index = sizes.indexOf(currentSize());
    Array.prototype.forEach.call(document.querySelectorAll('[data-action="font-decrease"]'), function (button) {
      button.disabled = index <= 0;
    });
    Array.prototype.forEach.call(document.querySelectorAll('[data-action="font-increase"]'), function (button) {
      button.disabled = index >= sizes.length - 1;
    });
    Array.prototype.forEach.call(document.querySelectorAll('[data-action="font-reset"]'), function (button) {
      button.disabled = index === 1;
    });
  }

  function applySize(size, persist, announce) {
    root.setAttribute('data-textsize', size);
    updateSizeButtons();
    if (announce && sizeStatus) {
      sizeStatus.textContent = 'Text size ' + sizeLabels[size];
    }
    if (persist) {
      storage.set(SIZE_KEY, size);
    }
  }

  function changeSize(direction) {
    var index = sizes.indexOf(currentSize());
    var next = Math.min(sizes.length - 1, Math.max(0, index + direction));
    if (next !== index) {
      applySize(sizes[next], true, true);
    }
  }

  applyTheme(currentTheme(), false);
  applySize(currentSize(), false, false);

  var sidebar = document.getElementById('book-sidebar');
  var tocToggle = document.getElementById('toc-toggle');
  var toc = document.getElementById('book-toc');
  var mobileQuery = window.matchMedia ? window.matchMedia('(max-width: 900px)') : null;

  function isMobile() {
    return mobileQuery ? mobileQuery.matches : window.innerWidth <= 900;
  }

  function setTocOpen(open) {
    if (!sidebar || !tocToggle) {
      return;
    }
    if (open) {
      sidebar.classList.remove('toc-collapsed');
    } else {
      sidebar.classList.add('toc-collapsed');
    }
    tocToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  if (tocToggle) {
    tocToggle.addEventListener('click', function () {
      var collapsed = sidebar ? sidebar.classList.contains('toc-collapsed') : false;
      setTocOpen(collapsed);
    });
  }

  if (toc) {
    toc.addEventListener('click', function (event) {
      var target = event.target;
      var link = target && target.closest ? target.closest('a') : null;
      if (link && isMobile()) {
        setTocOpen(false);
      }
    });
  }

  if (sidebar) {
    sidebar.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && isMobile() && !sidebar.classList.contains('toc-collapsed')) {
        setTocOpen(false);
        if (tocToggle) {
          tocToggle.focus();
        }
      }
    });
  }

  function handleViewportChange() {
    if (!isMobile()) {
      setTocOpen(true);
    }
  }

  if (mobileQuery) {
    if (mobileQuery.addEventListener) {
      mobileQuery.addEventListener('change', handleViewportChange);
    } else if (mobileQuery.addListener) {
      mobileQuery.addListener(handleViewportChange);
    }
  }

  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('#book-toc a[href^="#"]'));
  var linkEntries = [];
  var seenHeadings = [];

  tocLinks.forEach(function (link) {
    var href = link.getAttribute('href') || '';
    var id = decodeURIComponent(href.slice(1));
    if (!id) {
      return;
    }
    var target = document.getElementById(id);
    if (!target) {
      return;
    }
    var heading = /^H[1-6]$/.test(target.tagName) ? target : target.querySelector('h1, h2, h3, h4, h5, h6');
    if (!heading || seenHeadings.indexOf(heading) >= 0) {
      return;
    }
    seenHeadings.push(heading);
    linkEntries.push({ link: link, heading: heading });
  });

  function setActive(link) {
    tocLinks.forEach(function (item) {
      if (item === link) {
        item.setAttribute('aria-current', 'location');
      } else {
        item.removeAttribute('aria-current');
      }
    });
  }

  var observer = null;
  if ('IntersectionObserver' in window && linkEntries.length) {
    observer = new IntersectionObserver(function (entries) {
      var best = null;
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) {
          return;
        }
        var item = null;
        linkEntries.forEach(function (candidate) {
          if (candidate.heading === entry.target) {
            item = candidate;
          }
        });
        if (!item) {
          return;
        }
        if (!best || entry.boundingClientRect.top < best.entry.boundingClientRect.top) {
          best = { entry: entry, item: item };
        }
      });
      if (best) {
        setActive(best.item.link);
      }
    }, {
      rootMargin: '-12% 0px -70% 0px',
      threshold: [0, 1]
    });
    linkEntries.forEach(function (item) {
      observer.observe(item.heading);
    });
  }

  function updateActiveFromScroll() {
    var cutoff = window.innerHeight * 0.3;
    var chosen = linkEntries.length ? linkEntries[0] : null;
    linkEntries.forEach(function (item) {
      if (item.heading.getBoundingClientRect().top <= cutoff) {
        chosen = item;
      }
    });
    if (chosen) {
      setActive(chosen.link);
    }
  }

  if (!observer) {
    var activeTicking = false;
    var requestActive = function () {
      if (activeTicking) {
        return;
      }
      activeTicking = true;
      window.requestAnimationFrame(function () {
        updateActiveFromScroll();
        activeTicking = false;
      });
    };
    window.addEventListener('scroll', requestActive, { passive: true });
    window.addEventListener('resize', requestActive, { passive: true });
    updateActiveFromScroll();
  } else {
    window.setTimeout(updateActiveFromScroll, 0);
  }

  window.addEventListener('hashchange', function () {
    var hash = decodeURIComponent(window.location.hash.slice(1));
    linkEntries.forEach(function (item) {
      var section = item.heading.closest ? item.heading.closest('section') : null;
      if (item.heading.id === hash || (section && section.id === hash)) {
        setActive(item.link);
      }
    });
  });

  var progress = document.getElementById('reading-progress');
  var progressBar = document.getElementById('reading-progress-bar');
  var progressTicking = false;

  function updateProgress() {
    var doc = document.documentElement;
    var maximum = doc.scrollHeight - window.innerHeight;
    var top = window.pageYOffset || doc.scrollTop || 0;
    var percent = maximum > 0 ? Math.min(100, Math.max(0, (top / maximum) * 100)) : 0;
    if (progressBar) {
      progressBar.style.width = percent + '%';
    }
    if (progress) {
      progress.setAttribute('aria-valuenow', String(Math.round(percent)));
    }
  }

  function requestProgress() {
    if (progressTicking) {
      return;
    }
    progressTicking = true;
    window.requestAnimationFrame(function () {
      updateProgress();
      progressTicking = false;
    });
  }

  window.addEventListener('scroll', requestProgress, { passive: true });
  window.addEventListener('resize', requestProgress, { passive: true });
  window.addEventListener('load', requestProgress);
  updateProgress();

  document.addEventListener('click', function (event) {
    var target = event.target;
    var button = target && target.closest ? target.closest('[data-action]') : null;
    if (!button) {
      return;
    }
    var action = button.getAttribute('data-action');
    if (action === 'theme') {
      event.preventDefault();
      applyTheme(currentTheme() === 'dark' ? 'light' : 'dark', true);
    } else if (action === 'font-decrease') {
      event.preventDefault();
      changeSize(-1);
    } else if (action === 'font-reset') {
      event.preventDefault();
      applySize('normal', true, true);
    } else if (action === 'font-increase') {
      event.preventDefault();
      changeSize(1);
    } else if (action === 'print') {
      event.preventDefault();
      window.print();
    }
  });
}());
