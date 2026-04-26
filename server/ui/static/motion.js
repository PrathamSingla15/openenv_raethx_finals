/* TradeBench landing motion.
   Scroll-reveal (IntersectionObserver), hero halo parallax (rAF-throttled),
   and anchor-to-tab navigation. ~2.5 KB unminified. No deps. */

(() => {
    const REVEAL_SEL = '.tb-reveal';
    const HALO_SEL = '.tb-landing-halo';
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ── Scroll reveal ────────────────────────────────────────
    let io = null;
    if ('IntersectionObserver' in window) {
        io = new IntersectionObserver((entries) => {
            for (const e of entries) {
                if (!e.isIntersecting) continue;
                e.target.classList.add('is-in');
                io.unobserve(e.target);
            }
        }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    }

    function observeWithin(root) {
        if (!io || !root || !root.querySelectorAll) return;
        if (root.matches && root.matches(REVEAL_SEL) && !root.classList.contains('is-in')) {
            io.observe(root);
        }
        for (const el of root.querySelectorAll(REVEAL_SEL)) {
            if (!el.classList.contains('is-in')) io.observe(el);
        }
    }

    // ── Hero halo parallax ───────────────────────────────────
    function initParallax() {
        if (reduceMotion) return;
        const halo = document.querySelector(HALO_SEL);
        if (!halo) return;
        let raf = 0;
        const tick = () => {
            raf = 0;
            const y = window.scrollY || window.pageYOffset || 0;
            // Only commit transform while hero region is plausibly in view.
            if (y > window.innerHeight * 1.4) return;
            halo.style.transform = 'translate3d(0,' + (-y * 0.35).toFixed(2) + 'px,0)';
        };
        window.addEventListener('scroll', () => {
            if (!raf) raf = requestAnimationFrame(tick);
        }, { passive: true });
        tick();
    }

    // ── Tab-jump (anchor with data-tb-tab="<label>") ─────────
    function findTabButton(label) {
        if (!label) return null;
        const target = label.trim().toLowerCase();
        const buttons = document.querySelectorAll(
            '.tab-nav button, [role="tablist"] button, button[role="tab"]'
        );
        for (const b of buttons) {
            const txt = (b.innerText || b.textContent || '').trim().toLowerCase();
            if (txt === target) return b;
        }
        return null;
    }

    function smoothScrollTo(id) {
        if (!id) return false;
        const el = document.getElementById(id);
        if (!el) return false;
        el.scrollIntoView({
            behavior: reduceMotion ? 'auto' : 'smooth',
            block: 'start'
        });
        return true;
    }

    function attachClickHandlers() {
        document.body.addEventListener('click', (ev) => {
            const link = ev.target.closest('[data-tb-tab], a[href^="#"]');
            if (!link) return;

            const tab = link.getAttribute && link.getAttribute('data-tb-tab');
            const href = link.getAttribute('href') || '';
            const hashId = href.startsWith('#') ? href.slice(1) : '';

            if (tab) {
                ev.preventDefault();
                const btn = findTabButton(tab);
                if (btn && !btn.classList.contains('selected')) {
                    btn.click();
                }
                if (hashId) {
                    // Wait two frames for Gradio to mount the panel.
                    requestAnimationFrame(() => {
                        requestAnimationFrame(() => smoothScrollTo(hashId));
                    });
                }
                return;
            }

            if (hashId) {
                if (smoothScrollTo(hashId)) ev.preventDefault();
            }
        });
    }

    // ── Init ─────────────────────────────────────────────────
    function init() {
        observeWithin(document);
        initParallax();
        attachClickHandlers();

        if ('MutationObserver' in window && io) {
            const mo = new MutationObserver((muts) => {
                for (const m of muts) {
                    for (const n of m.addedNodes) {
                        if (n.nodeType === 1) observeWithin(n);
                    }
                }
            });
            mo.observe(document.body, { childList: true, subtree: true });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
