"""Helpers for embedding the canonical PNG figures from ``docs/figures/``.

The figures are mounted at ``/figures`` by ``server/app.py``; these helpers
render them inside the editorial ``.tb-figure`` card so headline / caption /
provenance live alongside the chart.
"""

from __future__ import annotations


def figure_card(
    *,
    src: str,
    title: str,
    meta: str = "",
    caption: str = "",
    dark: bool = False,
    alt: str = "",
) -> str:
    """Return the HTML for one ``.tb-figure`` card.

    Args:
        src: filename under /figures (e.g. ``reward_evolution.png``)
        title: small uppercase mono caption shown in the head row
        meta: right-aligned mono-uppercase provenance tag (model, iter, etc.)
        caption: prose caption shown in the foot of the card
        dark: render the image on the dark elevation surface instead of white
              (use this for figures that already have a transparent / dark bg)
        alt: alt text for the ``<img>`` tag; defaults to ``title``
    """

    body_class = "tb-figure-body is-dark" if dark else "tb-figure-body"
    img_alt = alt or title
    head = f"""
    <div class="tb-figure-head">
        <div class="tb-figure-title">{title}</div>
        <div class="tb-figure-meta">{meta}</div>
    </div>
    """ if (title or meta) else ""
    foot = f"""
    <div class="tb-figure-caption">{caption}</div>
    """ if caption else ""
    return f"""
<figure class="tb-figure">
    {head}
    <div class="{body_class}">
        <img src="/figures/{src}" alt="{img_alt}" loading="lazy" />
    </div>
    {foot}
</figure>
"""


__all__ = ["figure_card"]
