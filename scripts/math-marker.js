/**
 * 给启用了数学公式的文章加一个 `.has-jax` 标记。
 *
 * Stellar 的 MathJax 扩展只有在页面里出现 `.has-jax`（或 math/tex 脚本）时才会加载，
 * 而 hexo-renderer-marked 会把 `$...$` 原样留在 HTML 里，不会生成标记。
 * 这里按 Front Matter 的 `render.math` 精确标记，避免在无公式的页面加载 MathJax。
 */
'use strict';

hexo.extend.filter.register('after_post_render', function (data) {
  const math = data.render && data.render.math;
  if (!math || math === false) return data;
  if (typeof data.content !== 'string' || data.content.includes('class="has-jax"')) return data;
  data.content = '<span class="has-jax" hidden></span>' + data.content;
  return data;
});
