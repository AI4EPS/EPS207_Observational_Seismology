// MathJax for BOTH the markdown pages and the rendered notebooks. Notebook markdown is converted
// by nbconvert, not by the mkdocs markdown pipeline, so arithmatex alone does not reach it: this
// config lets MathJax find raw $...$ and $$...$$ anywhere on the page, while skipping code blocks.
window.MathJax = {
  tex: {
    inlineMath:  [["$", "$"],   ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};
// mkdocs-material swaps pages without a reload, so re-typeset on each navigation.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => { if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise(); });
}
